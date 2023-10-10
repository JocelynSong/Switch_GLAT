#!/bin/bash

src=$1
tgt=$2

export CUDA_VISIBLE_DEVICES=0
generation_path=./generation

mkdir -p generation

data_path=None 

model_path=./output
mkdir -p $model_path
remote_model_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/models 

python3 fairseq_cli/generate_mglat.py ${data_path} \
--task "multilingual_glat_translation" \
--dataset-impl "raw" \
--source-lang ${src} \
--target-lang ${tgt} \
--mt-steps ${src}-${tgt} \
--path output/checkpoint_${src}-${tgt}_best.pt:output/checkpoint_best.pt \
--iter-decode-with-external-reranker \
--batch-size 20 \
--iter-decode-with-beam 7 \
--max-len-a 1 \
--max-len-b 200 \
--results-path ${generation_path} \
--gen-subset valid \
--sacrebleu \
--skip-invalid-size-inputs-valid-test \
--remove-bpe