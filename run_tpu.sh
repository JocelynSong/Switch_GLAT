#!/bin/bash

src=de
tgt=en

export http_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080
export https_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080

python3 setup.py build_ext --inplace
pip install .
pip install sacremoses
pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.8.1-cp37-cp37m-linux_x86_64.whl

# Download and prepare the data
data_path=hdfs://harunava/home/byte_ailab_va_mlnlc/user/songzhenqiao/data/wmt_ten

local_root=.
output_path=${local_root}/output
mkdir -p ${output_path}
local_checkpoint_path=${output_path}/save_model
hdfs_checkpoint_path=hdfs://harunava/home/byte_ailab_va_mlnlc/user/songzhenqiao/model
remote_checkpoint_path=${hdfs_checkpoint_path}/transformer_${src}_${tgt}
mkdir -p ${local_checkpoint_path}
hadoop fs -mkdir -p ${hdfs_checkpoint_path}
hadoop fs -mkdir -p ${remote_checkpoint_path}

python3 fairseq_cli/train.py ${data_path} \
--save-dir ${local_checkpoint_path} \
--remote-save-dir ${remote_checkpoint_path} \
--task translation \
--source-lang ${src} \
--target-lang ${tgt} \
--dataset-impl "raw" \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--arch transformer \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--clip-norm 5 \
--lr 5e-4 \
--lr-scheduler inverse_sqrt \
--warmup-updates 10000 \
--dropout 0.1 \
--weight-decay 0.01 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 8192 \
--update-freq 1 \
--max-update 300000 \
--fp16 \
--tpu True \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 99999 \
--save-interval 99999 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--keep-interval-updates	10 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.0, "max_len_b": 200}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--activation-fn gelu \
--share-all-embeddings