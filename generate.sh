#!/bin/bash

src=$1
tgt=$2
rank=$3

export CUDA_VISIBLE_DEVICES=${rank}
generation_path=./generation/${src}_${tgt}/rank${rank}

mkdir -p ${generation_path}

data_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data/wmt_en_de_fr_ro_ru_zh/rank${rank}

model_path=./output/${src}_${tgt}
mkdir -p $model_path
remote_model_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/models/better_ten/transformer_big/${src}_${tgt}
hadoop fs -get $remote_model_path/checkpoint_best.pt $model_path

python3 fairseq_cli/generate.py ${data_path} \
--task multilingual_translation_song \
--dataset-impl "raw" \
--mt-steps ${src}-${tgt} \
--path ${model_path}/checkpoint_best.pt \
--batch-size 128 \
--beam 5 \
--nbest 5 \
--max-len-a 1 \
--max-len-b 200 \
--results-path ${generation_path} \
--gen-subset train \
--sacrebleu \
--skip-invalid-size-inputs-valid-test \
--remove-bpe

remote_path=$remote_model_path/$rank
hadoop fs -mkdir -p ${remote_path}
hadoop fs -put -f ${generation_path}/train.${src}-${tgt}.${src} $remote_path
hadoop fs -put -f ${generation_path}/train.${src}-${tgt}.${tgt} $remote_path