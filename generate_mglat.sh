#!/bin/bash

src=$1
tgt=$2

export CUDA_VISIBLE_DEVICES=0
generation_path=./generation

mkdir -p generation

data_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data/wmt_en_de_fr_ro_ru_zh/at_data/rank0
remote_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data/wmt_en_de_fr_ro_ru_zh/rank0
hadoop fs -mkdir -p ${remote_path}

model_path=./output
mkdir -p $model_path
remote_model_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/models/better_ten/many2many_all_shuffle/checkpoint_${src}-${tgt}_best.pt
hadoop fs -get $remote_model_path $model_path

python3 fairseq_cli/generate.py ${data_path} \
--task "multilingual_glat_translation" \
--dataset-impl "raw" \
--source-lang ${src} \
--target-lang ${tgt} \
--path output/checkpoint_${src}-${tgt}_best.pt \
--batch-size 8 \
--beam 5 \
--nbest 5 \
--max-len-a 1 \
--max-len-b 200 \
--results-path ${generation_path} \
--gen-subset valid \
--sacrebleu \
--skip-invalid-size-inputs-valid-test \
--remove-bpe


# hadoop fs -put -f ${generation_path}/${src}-${tgt}.txt $remote_path