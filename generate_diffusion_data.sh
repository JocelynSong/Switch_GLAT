#!/bin/bash

export http_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080
export https_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080

python3 setup.py build_ext --inplace
pip3 install .
pip3 install sacremoses

src=$1
tgt=$2
diffusion_lang=$3
rank=$4

export CUDA_VISIBLE_DEVICES=$rank

mkdir -p generation

data_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data/wmt_en_de_fr_ro_ru_zh/at_data/rank$rank

model_path=./output
mkdir -p $model_path
remote_model_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/models/multilingual_nat/vanilla_MNAT/checkpoint_${src}-${tgt}_best.pt
hadoop fs -get $remote_model_path $model_path


for rate in 0.2 0.3 0.4 0.5; do

remote_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data/wmt_en_de_fr_ro_ru_zh/nat_diffusion_data/rate${rate}/rank$rank
hadoop fs -mkdir -p ${remote_path}

local_save_path=./generation/rate${rate}/rank${rank}
mkdir -p ${local_save_path}

python3 fairseq_cli/generate_diffusion_data.py ${data_path} \
--task "multilingual_nat_translation" \
--dataset-impl "raw" \
--source-lang ${src} \
--target-lang ${tgt} \
--mt-steps ${src}-${tgt} \
--path output/checkpoint_${src}-${tgt}_best.pt \
--batch-size 1 \
--iter-decode-with-beam 7 \
--max-len-a 1 \
--max-len-b 200 \
--gen-subset train \
--sacrebleu \
--skip-invalid-size-inputs-valid-test \
--diffusion-num 300000 \
--diffusion-steps ${src}-${tgt}-${diffusion_lang} \
--diffusion-percentage ${rate} \
--diffusion-max-sentence 1 \
--output-translation-path ${local_save_path} \
--hdfs-save-path ${remote_path}

done


