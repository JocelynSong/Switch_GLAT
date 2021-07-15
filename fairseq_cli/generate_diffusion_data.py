#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import ast
import logging
import math
import os
import sys
from argparse import Namespace
from itertools import chain
import sacrebleu
import random

import numpy as np
import torch
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data.indexed_dataset import SampledIndexedRawTextDataset
from fairseq.data import LanguagePairDataset
from omegaconf import DictConfig


def sample_tokens_from_dataset(dataset, sample_num):
    samples = random.sample(range(len(dataset)), min(sample_num, len(dataset)))
    src_sent_lists = [dataset.src[i] for i in samples]
    tgt_sent_lists = [dataset.tgt[i] for i in samples]
    return src_sent_lists, tgt_sent_lists


def generate_diffused_data_for_each_pair(cfg, task, models, tgt_dict, pair, diffusion_lang, logger,
                                         use_cuda):
    logger.info("Generate diffusion data for language pair: %s" % pair)
    pair = cfg.task.mt_steps.split(",")[0]
    lgs = pair.split("-")
    src_lang, tgt_lang = lgs[0].strip(), lgs[1].strip()

    # construct dataset
    src_token_lists, tgt_token_lists = sample_tokens_from_dataset(task.get_dataset(cfg.dataset.gen_subset, pair),
                                                                  cfg.task.diffusion_num)

    sampled_src_dataset = SampledIndexedRawTextDataset(src_token_lists)
    sampled_tgt_dataset = SampledIndexedRawTextDataset(tgt_token_lists)

    sampled_pair_dataset = LanguagePairDataset(
        sampled_src_dataset,
        sampled_src_dataset.sizes,
        tgt_dict,
        sampled_tgt_dataset,
        sampled_tgt_dataset.sizes,
        tgt_dict,
        left_pad_source=cfg.task.left_pad_source,
        left_pad_target=cfg.task.left_pad_target,
        align_dataset=None,
        eos=None,
        num_buckets=0,
        shuffle=False,
        pad_to_multiple=1,
    )

    lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=sampled_pair_dataset,
        max_tokens=None,
        max_sentences=cfg.task.diffusion_max_sentence,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()

    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": cfg.generation.lm_weight}
    generator = task.build_generator(
        models, cfg.generation, extra_gen_cls_kwargs=extra_gen_cls_kwargs
    )

    output_path = os.path.join(cfg.task.output_translation_path, "diffusion.{}-{}".format(pair, diffusion_lang))
    src_path = output_path + ".src.txt"
    tgt_path = output_path + ".tgt.txt"
    fw_src = open(src_path, "w", encoding="utf-8")
    fw_tgt = open(tgt_path, "w", encoding="utf-8")

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        prefix_tokens = None
        if cfg.generation.prefix_size > 0:
            prefix_tokens = sample["target"][:, : cfg.generation.prefix_size]

        gen_timer.start()
        hypos = task.generate_diffusion_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            diffusion_lang=diffusion_lang,
            percentage=cfg.task.diffusion_percentage
        )

        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        for i, sample_id in enumerate(sample["id"].tolist()):
            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], tgt_dict.pad()
                )
            else:
                src_tokens = None

            src_str = tgt_dict.diffusion_string(src_tokens, cfg.common_eval.post_process)
            fw_tgt.write(src_str + "\n")

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: 1]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=None,
                    tgt_dict=tgt_dict,
                    remove_bpe=None,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                fw_src.write(hypo_str + "\n")

        wps_meter.update(num_generated_tokens)
        progress.log({"wps": round(wps_meter.avg)})

    logger.info("Done: generating diffusion data")
    fw_src.close()
    fw_tgt.close()

    os.system("hadoop fs -put -f {} {}".format(src_path, cfg.task.hdfs_save_path))
    os.system("hadoop fs -put -f {} {}".format(tgt_path, cfg.task.hdfs_save_path))


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for generation!"
    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        cfg.generation.replace_unk is None or cfg.dataset.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)
    else:
        return _main(cfg, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(cfg: DictConfig, output_file):
    refs, gens = [], []

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(cfg.common)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 12000
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    # Load dataset splits
    task = tasks.setup_task(cfg.task)

    logger.info("Load training data!")
    for pair in cfg.task.mt_steps.split(","):
        task.load_para_dataset(cfg.dataset.gen_subset, pair, epoch=1)

    # Set dictionaries
    try:
        src_dict = getattr(task, "source_dictionary", None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # generate diffusion data for each language pair
    for diffusion_step in cfg.task.diffusion_steps.strip().split(","):
        langs = diffusion_step.split("-")
        src_lang, tgt_lang, diffusion_lang = langs[0].strip(), langs[1].strip(), langs[2].strip()

        generate_diffused_data_for_each_pair(cfg, task, models, tgt_dict, "-".join([src_lang, tgt_lang]),
                                             diffusion_lang, logger, use_cuda)


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
