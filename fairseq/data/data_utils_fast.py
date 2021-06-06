# cython: language_level=3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

DTYPE = np.int64
# ctypedef np.int64_t DTYPE_t


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if (max_sentences > 0 and len(batch) == max_sentences):
        return 1
    if (max_tokens > 0 and num_tokens > max_tokens):
        return 1
    return 0


def batch_by_size_fast(indices, num_tokens_fn, max_tokens, max_sentences,
                       bsz_mult):
    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    indices_view = indices

    for i in range(len(indices_view)):
        idx = indices_view[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert max_tokens <= 0 or sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches
