#!/bin/bash

src=$1
tgt=$2
rank=$3

export CUDA_VISIBLE_DEVICES=${rank}
generation_path=./en_pt/rank${rank}

mkdir -p ${generation_path}

data_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data/ted_ten/new_data/rank0

model_path=./output/${src}_${tgt}
mkdir -p $model_path
remote_model_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/models/ted/transformer_base/${src}_${tgt}/checkpoint_best.pt
hadoop fs -get $remote_model_path $model_path

python3 fairseq_cli/generate.py ${data_path} \
--task multilingual_translation_song \
--dataset-impl "raw" \
--mt-steps ${src}-${tgt} \
--path output/${src}_${tgt}/checkpoint_best.pt \
--batch-size 32 \
--beam 5 \
--nbest 5 \
--max-len-a 1 \
--max-len-b 200 \
--results-path ${generation_path} \
--gen-subset train \
--sacrebleu \
--skip-invalid-size-inputs-valid-test \
--remove-bpe


# hadoop fs -put -f ${generation_path}/${src}-${tgt}.txt $remote_path