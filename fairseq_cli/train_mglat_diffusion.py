#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
import random
from typing import Dict, Optional, Any, List, Tuple, Callable
import json

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators, data_utils
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


starting_pair_ratio = {"en-zh": 2513.6, "zh-en": 2513.6, "en-de": 456.2, "de-en": 456.2, "en-fr": 4084.2,
                       "fr-en": 4084.2, "en-ro": 62.2, "ro-en": 62.2, "en-ru": 258.9, "ru-en": 258.9}


def get_ratio_list(starting_ratio_list, T):
    alpha = 1.0 / T
    new_list = np.power(starting_ratio_list, alpha)
    new_list = new_list / np.sum(new_list)
    new_list[-1] = 1.0 - np.sum(new_list[: -1])
    return new_list


def get_diffused_epoch_iter(cfg, task, model, diffusion_step, epoch):
    return task.get_batch_iterator(
            dataset=task.get_diffusion_dataset(diffusion_step),
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                model.max_positions(),
                cfg.dataset.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=1,
            shard_id=0,
            num_workers=cfg.dataset.num_workers,
            epoch=epoch,
            data_buffer_size=cfg.dataset.data_buffer_size,
            disable_iterator_cache=False)


def get_back_trans_epoch_iter(cfg, task, model, back_pair, epoch):
    return task.get_batch_iterator(
        dataset=task.get_back_trans_dataset(back_pair),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
            cfg.dataset.max_tokens,
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=1,
        shard_id=0,
        num_workers=cfg.dataset.num_workers,
        epoch=epoch,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )


def get_diffused_iter_dict(cfg, task, model, epoch, update_freq, diffusion_path):
    diffused_epoch_itr, diffused_iter_dict = {}, {}
    diffusion_cand = {}
    for diffusion_step in cfg.task.diffusion_steps.strip().split(","):
        langs = diffusion_step.split("-")
        src_lang, tgt_lang, diffusion_lang = langs[0].strip(), langs[1].strip(), langs[2].strip()
        diffusion_cand["-".join([src_lang, tgt_lang])] = diffusion_step

        # load diffusion data
        task.load_diffusion_dataset(diffusion_path, diffusion_step)
        diffused_epoch_itr[("mt", src_lang, tgt_lang, diffusion_lang)] = get_diffused_epoch_iter(cfg, task, model, diffusion_step,
                                                                                                 epoch)

        diffused_iter_dict[("mt", src_lang, tgt_lang, diffusion_lang)] = \
            diffused_epoch_itr[("mt", src_lang, tgt_lang, diffusion_lang)].next_epoch_itr(
                fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
                shuffle=True)

        diffused_iter_dict[("mt", src_lang, tgt_lang, diffusion_lang)] = iterators.GroupedIterator(
            diffused_iter_dict[("mt", src_lang, tgt_lang, diffusion_lang)],
            update_freq)
    task.diffusion_cand = diffusion_cand
    return diffused_epoch_itr, diffused_iter_dict


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    local_rank = cfg.distributed_training.device_id
    cfg.task.data = cfg.task.data.split(",")
    cfg.task.data = os.path.join(cfg.task.data[0], "rank{}".format(local_rank))

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    # data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    for pair in cfg.task.mt_steps.split(","):
        task.load_para_dataset("valid", pair, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    logger.info("loading training data")
    for pair in cfg.task.mt_steps.split(","):
        logger.info("load parallel data: %s pair" % pair)
        task.load_para_dataset(cfg.dataset.train_subset, pair, epoch=0, buffer_size=cfg.task.buffer_size,
                               enable_lazy_load=cfg.task.enable_lazy_loader)

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("load_checkpoint")  # wait for all workers

    logger.info("Set iterator!")
    update_freq = (
        cfg.optimization.update_freq[epoch_itr["epoch"] - 1]
        if epoch_itr["epoch"] <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )

    pair_list = cfg.task.mt_steps.split(",")
    iter_dict = {}
    for pair in pair_list:
        langs = pair.split("-")
        src_lang, tgt_lang = langs[0].strip(), langs[1].strip()

        iter_dict[("mt", src_lang, tgt_lang)] = epoch_itr[("mt", src_lang, tgt_lang)].next_epoch_itr(
            fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
            shuffle=(epoch_itr["epoch"] > cfg.dataset.curriculum),
        )
        iter_dict[("mt", src_lang, tgt_lang)] = iterators.GroupedIterator(iter_dict[("mt", src_lang, tgt_lang)],
                                                                          update_freq)
        if cfg.common.tpu:
            iter_dict[("mt", src_lang, tgt_lang)] = utils.tpu_data_loader(iter_dict[("mt", src_lang, tgt_lang)])

    logger.info("loading back translation data and setting iterator!")
    back_epoch_itr, back_iter_dict = {}, {}
    if cfg.task.enable_back_translation and not cfg.task.enable_lazy_loader:
        for back_pair in cfg.task.back_translation_steps.split(","):
            back_langs = back_pair.split("-")
            back_src, back_tgt = back_langs[0].strip(), back_langs[1].strip()

            # load back-translation data
            back_translation_path = os.path.join(cfg.task.back_translation_path, "rank{}".format(local_rank))
            task.load_back_translation_dataset(back_translation_path, back_pair)
            back_epoch_itr[("mt", back_src, back_tgt)] = get_back_trans_epoch_iter(cfg, task, model, back_pair,
                                                                                   epoch=epoch_itr["epoch"])

            back_iter_dict[("mt", back_src, back_tgt)] = back_epoch_itr[("mt", back_src, back_tgt)].next_epoch_itr(
                    fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
                    shuffle=True)
            back_iter_dict[("mt", back_src, back_tgt)] = iterators.GroupedIterator(
                back_iter_dict[("mt", back_src, back_tgt)], update_freq)

            if cfg.common.tpu:
                back_iter_dict[("mt", back_src, back_tgt)] = utils.tpu_data_loader(back_iter_dict[("mt", back_src,
                                                                                                   back_tgt)])

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()

    best_valid_bleu = json.loads(cfg.task.vanilla_model_bleu)
    best_valid_bleu['avg'] = -1
    pair_list = cfg.task.mt_steps.split(",")

    ratio_list_level1 = None
    # ratio_list_level2 = [0.0833, 0.0833, 0.1612, 0.1612, 0.0461, 0.0461, 0.0703, 0.0703, 0.1391, 0.1391]
    sample_pair_list = [starting_pair_ratio[k] for k in pair_list]
    ratio_list_level2 = [k / sum(sample_pair_list) for k in sample_pair_list]
    ratio_list_level2 = get_ratio_list(ratio_list_level2, 3.33)

    diffusion_rate = ["rate0.2", "rate0.3", "rate0.4", "rate0.5"]
    diffused_epoch_itr, diffused_iter_dict = {}, {}
    schedule_epoch = 0
    lazy_load_epoch = 0

    while epoch_itr["epoch"] <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # update training data piece for lazy loader
        if cfg.task.enable_lazy_loader:
            if (schedule_epoch != 0) and (epoch_itr["epoch"] % cfg.task.lazy_load_interval == 0):
                for pair in pair_list:
                    logger.info("Loading new piece of data for pair %s" % pair)
                    task.load_para_dataset(cfg.dataset.train_subset, pair, epoch=lazy_load_epoch,
                                           buffer_size=cfg.task.buffer_size,
                                           enable_lazy_load=cfg.task.enable_lazy_loader)
                epoch_itr = trainer.get_train_iterator(epoch=epoch_itr["epoch"])

                for pair in pair_list:
                    langs = pair.split("-")
                    src_lang, tgt_lang = langs[0].strip(), langs[1].strip()
                    iter_dict[("mt", src_lang, tgt_lang)] = epoch_itr[("mt", src_lang, tgt_lang)].next_epoch_itr(
                        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
                        shuffle=(epoch_itr["epoch"] > cfg.dataset.curriculum),
                    )
                    iter_dict[("mt", src_lang, tgt_lang)] = iterators.GroupedIterator(
                        iter_dict[("mt", src_lang, tgt_lang)], update_freq)
                    if cfg.common.tpu:
                        iter_dict[("mt", src_lang, tgt_lang)] = utils.tpu_data_loader(
                            iter_dict[("mt", src_lang, tgt_lang)])

            if (schedule_epoch == 0) or (epoch_itr["epoch"] % cfg.task.lazy_load_interval == 0):
                for back_pair in cfg.task.back_translation_steps.split(","):
                    back_langs = back_pair.split("-")
                    back_src, back_tgt = back_langs[0].strip(), back_langs[1].strip()

                    # load back-translation data
                    back_translation_path = os.path.join(cfg.task.back_translation_path, "rank{}".format(local_rank))
                    task.load_back_translation_dataset(back_translation_path, back_pair, lazy_load_epoch,
                                                       cfg.task.buffer_size, cfg.task.enable_lazy_loader)
                    back_epoch_itr[("mt", back_src, back_tgt)] = get_back_trans_epoch_iter(cfg, task, model, back_pair,
                                                                                           epoch=epoch_itr["epoch"])

                    back_iter_dict[("mt", back_src, back_tgt)] = \
                        back_epoch_itr[("mt", back_src, back_tgt)].next_epoch_itr(
                            fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
                            shuffle=True)
                    back_iter_dict[("mt", back_src, back_tgt)] = iterators.GroupedIterator(
                        back_iter_dict[("mt", back_src, back_tgt)], update_freq)

                    if cfg.common.tpu:
                        back_iter_dict[("mt", back_src, back_tgt)] = utils.tpu_data_loader(
                            back_iter_dict[("mt", back_src,
                                            back_tgt)])
                lazy_load_epoch += 1

        # load diffusion data
        if schedule_epoch % cfg.task.diffusion_generation_interval == 0:
            index = int(schedule_epoch / cfg.task.diffusion_generation_interval) % 4
            diffusion_path = os.path.join(cfg.task.diffusion_data_path, diffusion_rate[index],
                                          "rank{}".format(local_rank))
            diffused_epoch_itr, diffused_iter_dict = get_diffused_iter_dict(cfg, task, model, epoch_itr["epoch"], update_freq,
                                                                            diffusion_path)

        # set sampling ratio
        ratio_list = ratio_list_level2

        # train for one epoch
        valid_losses, should_stop, best_valid_bleu = train(cfg, trainer, task, epoch_itr, iter_dict, pair_list,
                                                           best_valid_bleu, ratio_list, back_epoch_itr, back_iter_dict,
                                                           diffused_epoch_itr, diffused_iter_dict, schedule_epoch)

        logger.info("Save original checkpoints")
        langs = cfg.task.metric_pair.split("-")
        src_lang, tgt_lang = langs[0].strip(), langs[1].strip()
        # save every end of epoch
        checkpoint_utils.save_checkpoint(cfg.checkpoint, trainer, epoch_itr[("mt", src_lang, tgt_lang)],
                                         valid_losses[cfg.task.metric_pair])

        epoch_itr["epoch"] += 1
        schedule_epoch += 1

        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr["epoch"], valid_losses[cfg.task.metric_pair])

    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, iter_dict, pair_list, best_valid_bleu,
        ratio_list, back_epoch_itr, back_iter_dict, diffused_epoch_itr, diffused_iter_dict, schedule_epoch) \
        -> Tuple[dict, bool, dict]:
    """
    Train the model for one epoch and return validation losses.
    """
    # Initialize data iterator
    update_freq = (
        cfg.optimization.update_freq[epoch_itr["epoch"] - 1]
        if epoch_itr["epoch"] <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )

    trainer.begin_epoch(epoch_itr["epoch"])

    valid_subsets = cfg.dataset.valid_subset.split(",")
    logger.info("Start iterating over samples for epoch %d" % epoch_itr["epoch"])

    trainer.step_size = 0
    back_translation_steps = cfg.task.back_translation_steps.split(",")
    diffusion_steps = cfg.task.diffusion_steps.split(",")  # [en-de-fr, en-fr-de]

    while trainer.step_size < cfg.dataset.validate_after_updates:
        start_num_updates = trainer.get_num_updates()
        annealed_dropout = (max(0., 1. - float(start_num_updates) / cfg.task.annealing_total_num)) * cfg.model.dropout
        trainer.get_model().encoder.update_dropout_rate(annealed_dropout)
        trainer.get_model().decoder.update_dropout_rate(annealed_dropout)

        # train one step for diffusion data
        if trainer.step_size % cfg.task.diffusion_interval == 0:
            pair = random.choice(diffusion_steps)
            langs = pair.split("-")
            src_lang, tgt_lang, diffusion_lang = langs[0].strip(), langs[1].strip(), langs[2].strip()
            try:
                samples = next(diffused_iter_dict[("mt", src_lang, tgt_lang, diffusion_lang)])
            except StopIteration:
                new_diffusion_iter_epoch = diffused_epoch_itr[("mt", src_lang, tgt_lang, diffusion_lang)].epoch + 1
                del diffused_epoch_itr[("mt", src_lang, tgt_lang, diffusion_lang)]
                del diffused_iter_dict[("mt", src_lang, tgt_lang, diffusion_lang)]

                diffused_epoch_itr[("mt", src_lang, tgt_lang, diffusion_lang)] = get_diffused_epoch_iter(
                    cfg, task, trainer.get_model(), pair, new_diffusion_iter_epoch)
                diffused_iter_dict[("mt", src_lang, tgt_lang, diffusion_lang)] = \
                    diffused_epoch_itr[("mt", src_lang, tgt_lang, diffusion_lang)].next_epoch_itr(
                    fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
                    shuffle=True)
                diffused_iter_dict[("mt", src_lang, tgt_lang, diffusion_lang)] = \
                    iterators.GroupedIterator(diffused_iter_dict[("mt", src_lang, tgt_lang, diffusion_lang)], update_freq)
                samples = next(diffused_iter_dict[("mt", src_lang, tgt_lang, diffusion_lang)])
        elif cfg.task.enable_back_translation and trainer.step_size % cfg.task.back_translation_interval == 0:
            pair = random.choice(back_translation_steps)
            langs = pair.split("-")
            src_lang, tgt_lang = langs[0].strip(), langs[1].strip()
            try:
                samples = next(back_iter_dict[("mt", src_lang, tgt_lang)])
            except StopIteration:
                new_data_iter_epoch = back_epoch_itr[("mt", src_lang, tgt_lang)].epoch + 1
                del back_epoch_itr[("mt", src_lang, tgt_lang)]
                del back_iter_dict[("mt", src_lang, tgt_lang)]

                back_epoch_itr[("mt", src_lang, tgt_lang)] = get_back_trans_epoch_iter(cfg, task, trainer.get_model(), pair, new_data_iter_epoch)
                back_iter_dict[("mt", src_lang, tgt_lang)] = back_epoch_itr[("mt", src_lang, tgt_lang)].next_epoch_itr(
                    fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus, shuffle=True)
                back_iter_dict[("mt", src_lang, tgt_lang)] = \
                    iterators.GroupedIterator(back_iter_dict[("mt", src_lang, tgt_lang)], update_freq)
                samples = next(back_iter_dict[("mt", src_lang, tgt_lang)])
        else:
            if ratio_list is not None:
                pair = np.random.choice(pair_list, 1, p=ratio_list)[0]
            else:
                pair = random.choice(pair_list)
            langs = pair.split("-")
            src_lang, tgt_lang = langs[0].strip(), langs[1].strip()
            try:
                samples = next(iter_dict[("mt", src_lang, tgt_lang)])
            except StopIteration:
                new_data_iter_epoch = epoch_itr[("mt", src_lang, tgt_lang)].epoch + 1
                del epoch_itr[("mt", src_lang, tgt_lang)]
                del iter_dict[("mt", src_lang, tgt_lang)]

                epoch_itr[("mt", src_lang, tgt_lang)] = trainer.get_single_pair_train_iterator(pair, new_data_iter_epoch)
                iter_dict[("mt", src_lang, tgt_lang)] = epoch_itr[("mt", src_lang, tgt_lang)].next_epoch_itr(
                    fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
                    shuffle=(epoch_itr["epoch"] > cfg.dataset.curriculum))
                iter_dict[("mt", src_lang, tgt_lang)] = iterators.GroupedIterator(iter_dict[("mt", src_lang, tgt_lang)],
                                                                                  update_freq)
                samples = next(iter_dict[("mt", src_lang, tgt_lang)])

        for sample in samples:
            sample["src_lang"] = src_lang
            sample["tgt_lang"] = tgt_lang

        with metrics.aggregate("train_inner"):
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))  # starting from here
                info = "Epoch=%d, Step=%d, Updates=%d" % (epoch_itr["epoch"], trainer.step_size + 1,
                                                          trainer.get_num_updates())
                if trainer.step_size % cfg.task.diffusion_interval == 0:
                    info = "%s, Diffusion data steps: %s" % (info, pair)
                elif cfg.task.enable_back_translation and trainer.step_size % cfg.task.back_translation_interval == 0:
                    info = "%s, Back translation steps: %s" % (info, pair)
                else:
                    info = "%s, KD data steps: %s" % (info, pair)
                for key in stats.keys():
                    info = "{} |{}: {}".format(info, key, stats[key])
                logger.info(info)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        trainer.step_size += 1

    end_of_epoch = True
    valid_losses, should_stop, best_valid_bleu = validate_and_save(
        cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch, best_valid_bleu, pair_list
    )

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr["epoch"]))
    # stats = get_training_stats(metrics.get_smoothed_values("train"))
    info = "Epoch=%d, Valid BLEU " % (epoch_itr["epoch"])
    for pair in valid_losses.keys():
        info = "{} |{}: {}".format(info, pair, valid_losses[pair])
    logger.info(info)

    info = "Epoch=%d, Best Valid BLEU " % (epoch_itr["epoch"])
    for pair in best_valid_bleu.keys():
        info = "{} |{}: {}".format(info, pair, best_valid_bleu[pair])
    logger.info(info)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop, best_valid_bleu


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    best_valid_bleu,
    pair_list
) -> Tuple[dict, bool, dict]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = True
    do_validate = True

    # Validate
    valid_losses = dict()
    total_bleu, avg_bleu = 0.0, 0.0

    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets, pair_list)

        for pair in pair_list:
            total_bleu += valid_losses[pair]
            if valid_losses[pair] > best_valid_bleu[pair]:
                best_valid_bleu[pair] = valid_losses[pair]
        avg_bleu = total_bleu / len(pair_list)
        valid_losses['avg'] = avg_bleu
        if avg_bleu > best_valid_bleu['avg']:
            best_valid_bleu['avg'] = avg_bleu

    should_stop |= should_stop_early(cfg, valid_losses[cfg.task.metric_pair])

    # Save checkpoint
    if do_save or should_stop:
        for pair in pair_list:
            langs = pair.split("-")
            src_lang, tgt_lang = langs[0].strip(), langs[1].strip()
            checkpoint_utils.save_checkpoint_multilingual(cfg.checkpoint, trainer,
                                                          epoch_itr[("mt", src_lang, tgt_lang)],
                                                          valid_losses[pair], pair, epoch_itr["epoch"])

        if cfg.task.metric_pair != "":
            langs = cfg.task.metric_pair.split("-")
            src_lang, tgt_lang = langs[0].strip(), langs[1].strip()
            checkpoint_utils.save_checkpoint_multilingual(cfg.checkpoint, trainer,
                                                          epoch_itr[("mt", src_lang, tgt_lang)],
                                                          avg_bleu, 'avg', epoch_itr["epoch"])

    return valid_losses, should_stop, best_valid_bleu


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
    pair_list
):
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr["epoch"])
    valid_losses = dict()
    subset = subsets[0]
    for pair in pair_list:
        logger.info('begin validation on valid subset for pair'.format(pair))
        langs = pair.split("-")
        src_lang, tgt_lang = langs[0].strip(), langs[1].strip()

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset, pair).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr["epoch"],
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break

                sample["src_lang"] = src_lang
                sample["tgt_lang"] = tgt_lang
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
        stats["pair"] = pair

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses[pair] = stats[cfg.checkpoint.best_checkpoint_metric]
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
