# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from dataclasses import dataclass, field
from math import log
import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset, BufferedIndexedRawTextDataset, IndexedRawTextDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask, load_langpair_dataset
from fairseq.utils import new_arange
import logging
from omegaconf import II
import numpy as np


NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])


@dataclass
class MultilingualGlatTranslationConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="full_mask",
        metadata={
            "help": "type of noise"
        })
    total_sample_updates: int = field(
        default=300000,
        metadata={"help": "Total updates number for mglat glancing sampling"}
    )
    minus_p: float = field(
        default=0.2,
        metadata={"help": "Starting sampling ratio for mglat training"}
    )
    mt_steps: str = field(
        default="de-en,en-de,fr-en,en-fr",
        metadata={"help": "multilingual machine translation steps"}
    )
    metric_pair: str = field(
        default="de-en",
        metadata={"help": "metric language pair for early stop"}
    )
    annealing_total_num: int = field(
        default=1200000,
        metadata={"help": "Total number of dropout annealing"}
    )
    diffusion_num: int = field(
        default=300000,
        metadata={"help": "the number of generated diffusion data"}
    )
    diffusion_steps: str = field(
        default="de-en-fr",
        metadata={"help": "diffusion steps for the second stage training"}
    )
    diffusion_percentage: float = field(
        default=0.1,
        metadata={"help": "masked rate for the diffused words"}
    )
    diffusion_max_sentence: int = field(
        default=8,
        metadata={"help": "batch size for generating diffusion sentence"}
    )
    diffusion_length_beam: int = field(
        default=7,
        metadata={"help": "length beam for generating diffusion sentence"}
    )
    output_translation_path: str = field(
        default="",
        metadata={"help": "output path for generated diffusion data"}
    )
    hdfs_save_path: str = field(
        default="",
        metadata={"help": "Saved hdfs path for generated diffusion data"}
    )
    diffusion_data_path: str = field(
        default="",
        metadata={"help": "Data path for diffusion data"}
    )
    diffusion_generation_interval: int = field(
        default=50,
        metadata={"help": "Frequency for changing a diffusion data split with a different diffusion rsate"}
    )
    diffusion_interval: int = field(
        default=5,
        metadata={"help": "train a duffusion step every N steps"}
    )
    vanilla_model_bleu: str = field(
        default="",
        metadata={"help": "validation bleu scores for vanilla model, based on which mglat will be trained continuously"}
    )
    enable_back_translation: bool = field(
        default=False,
        metadata={"help": "whether use the back-translation dara in training"}
    )
    back_translation_steps: str = field(
        default="",
        metadata={"help": "training steps for back-translation"}
    )
    back_translation_path: str = field(
        default="",
        metadata={"help": "back translation data path"}
    )
    back_translation_interval: int = field(
        default=4,
        metadata={"help": "run one back_translation step every N steps"}
    )
    enable_lazy_loader: bool = field(
        default=False,
        metadata={"help": "whether use the lazy data loader"}
    )
    buffer_size: int = field(
        default=1000000,
        metadata={"help": "buffer size for using lazy data loader"}
    )
    lazy_load_interval: int = field(
        default=30,
        metadata={"help": "interval for loading another buffered data"}
    )


logger = logging.getLogger(__name__)


@register_task("multilingual_nat_translation", dataclass=MultilingualGlatTranslationConfig)
class MultilingualNATTranslationTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    cfg: MultilingualGlatTranslationConfig

    def __init__(self, cfg: MultilingualGlatTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.datasets["para"] = dict()
        self.datasets["para"]["train"], self.datasets["para"]["valid"], self.datasets["para"]["test"] = {}, {}, {}
        self.datasets["diffusion"], self.datasets["back_trans"] = {}, {}
        # data iterators
        self.iterator = {}

    @classmethod
    def setup_task(cls, cfg: MultilingualGlatTranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], "dict.txt"))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], "dict.txt"))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        logger.info("Src dictionary: {} types".format(len(src_dict)))
        logger.info("Tgt dictionary: {} types".format(len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_para_dataset(self, split, pair, epoch=1, combine=False, buffer_size=1000000, enable_lazy_load=False,
                          **kwargs):
        """
        Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[0]

        # compute langcode
        langs = pair.split("-")
        src, tgt = langs[0].strip(), langs[1].strip()

        self.datasets["para"][split][pair] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
            epoch=epoch,
            buffer_size=buffer_size,
            enable_lazy_load=enable_lazy_load
        )

    def load_back_translation_dataset(self, back_translation_path, pair, epoch=1, buffer_size=1000000,
                                      enable_lazy_load=False):
        """
        Load a given dataset split.

        Args:
             back_translation_path(str): path of back-translation data
        """
        def get_indexed_dataset(path, cur_epoch, back_buffer_size, dictionary):
            if enable_lazy_load:
                return BufferedIndexedRawTextDataset(path, cur_epoch, back_buffer_size)
            else:
                return IndexedRawTextDataset(path, dictionary)

        langs = pair.split("-")
        src_lang, tgt_lang = langs[0].strip(), langs[1].strip()

        prefix = os.path.join(back_translation_path, "back.{}.".format(pair))
        src_dataset = get_indexed_dataset(prefix + src_lang + ".pth", epoch, buffer_size, self.src_dict)
        tgt_dataset = get_indexed_dataset(prefix + tgt_lang + ".pth", epoch, buffer_size, self.tgt_dict)

        logger.info("loading back translation data for pair %s: %d" % (pair, len(src_dataset)))
        self.datasets["back_trans"][pair] = LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            self.src_dict,
            tgt_dataset,
            tgt_dataset.sizes,
            self.tgt_dict,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            shuffle=True)

    def load_diffusion_dataset(self, diffusion_path, diffusion_step):
        """
        Load a given dataset split.

        Args:
             diffusion_path(str): path of back-translation data
             diffusion_step:
        """
        def get_indexed_dataset(path, dictionary):
            return IndexedRawTextDataset(path, dictionary)

        prefix = os.path.join(diffusion_path, "diffusion.{}.".format(diffusion_step))
        src_dataset = get_indexed_dataset(prefix + "src.pth", self.src_dict)
        tgt_dataset = get_indexed_dataset(prefix + "tgt.pth", self.tgt_dict)

        logger.info("loading diffusion data for step %s: %d" % (diffusion_step, len(src_dataset)))

        self.datasets["diffusion"][diffusion_step] = LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            self.src_dict,
            tgt_dataset,
            tgt_dataset.sizes,
            self.tgt_dict,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            shuffle=True)

    def get_multilingual_batch_iterator(
        self,
        split,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            split (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        for pair in self.cfg.mt_steps.split(","):
            logger.info("get iterator for pair %s" % pair)

            batch_iterator = self.get_batch_iterator(
                dataset=self.datasets["para"][split][pair],
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
            )
            langs = pair.split("-")
            src_lang, tgt_lang = langs[0].strip(), langs[1].strip()
            self.iterator[("mt", src_lang, tgt_lang)] = batch_iterator
        self.iterator["epoch"] = epoch
        return self.iterator

    def get_single_pair_batch_iterator(
        self,
        pair,
        split,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            pair: language pair to create a new iterator
            split (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        logger.info("Run end of data iterator! Get iterator for pair %s" % pair)

        batch_iterator = self.get_batch_iterator(
            dataset=self.datasets["para"][split][pair],
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            data_buffer_size=data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        )
        langs = pair.split("-")
        src_lang, tgt_lang = langs[0].strip(), langs[1].strip()
        self.iterator[("mt", src_lang, tgt_lang)] = batch_iterator
        return batch_iterator

    def get_dataset(self, split, pair):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            pair: specific language or data-pair

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset
        if split not in self.datasets["para"]:
            raise KeyError('{} Dataset not loaded: {}'.format("para", split))
        if not isinstance(self.datasets["para"][split][pair], FairseqDataset):
            raise TypeError('Datasets are expected to be of type FairseqDataset')
        return self.datasets["para"][split][pair]

    def get_back_trans_dataset(self, pair):
        """
        Return a loaded back-translation dataset split.

        Args:
            pair: specific language or data-pair

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset
        if pair not in self.datasets["back_trans"]:
            raise KeyError('{} Dataset not loaded: {}'.format("back-translation", pair))
        if not isinstance(self.datasets["back_trans"][pair], FairseqDataset):
            raise TypeError('Datasets are expected to be of type FairseqDataset')
        return self.datasets["back_trans"][pair]

    def get_diffusion_dataset(self, diffusion_step):
        """
        Return a loaded diffusion dataset split.

        Args:
            diffusion_step: specific diffusion data to load

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset
        if diffusion_step not in self.datasets["diffusion"]:
            raise KeyError('{} Dataset not loaded: {}'.format("diffusion", diffusion_step))
        if not isinstance(self.datasets["diffusion"][diffusion_step], FairseqDataset):
            raise TypeError('Datasets are expected to be of type FairseqDataset')
        return self.datasets["diffusion"][diffusion_step]

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import MultilingualIterativeRefinementGenerator

        return MultilingualIterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        src_lang = sample["src_lang"]
        tgt_lang = sample["tgt_lang"]

        model.train()
        sample["prev_target"] = self.inject_noise(sample["target"])

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample, src_lang=src_lang, tgt_lang=tgt_lang)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, src_lang=None, tgt_lang=None):
        src_lang = sample["src_lang"]
        tgt_lang = sample["tgt_lang"]

        model.eval()
        with torch.no_grad():
            sample["prev_target"] = self.inject_noise(sample["target"])
            loss, sample_size, logging_output = criterion(model, sample, src_lang=src_lang, tgt_lang=tgt_lang)
            EVAL_BLEU_ORDER = 4
            if self.cfg.eval_bleu:
                bleu = self.inference_with_bleu(self.sequence_generator, sample, model, src_lang, tgt_lang)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def inference_with_bleu(self, generator, sample, model, src_lang, tgt_lang):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None, src_lang=src_lang,
                                      tgt_lang=tgt_lang)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, src_lang=None, tgt_lang=None,
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, src_lang=src_lang, tgt_lang=tgt_lang
            )

    def generate_diffusion_step(
        self, generator, models, sample, prefix_tokens=None, src_lang=None, tgt_lang=None, diffusion_lang=None,
            percentage=0.1):
        with torch.no_grad():
            return generator.generate_diffusion_data(
                models, sample, prefix_tokens=prefix_tokens, src_lang=src_lang, tgt_lang=tgt_lang,
                diffusion_lang=diffusion_lang, percentage=percentage
            )
