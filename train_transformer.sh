#!/bin/bash

src=$1
tgt=$2

cd /opt/tiger/lab/mlnlc/faster_multilingual_glat

source bashutil.sh

auto_install_dependencies

# Download and prepare the data
data_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data/wmt_en_de_fr_ro_ru_zh

local_root=.
output_path=${local_root}/output
mkdir -p ${output_path}
local_checkpoint_path=${output_path}/save_model
hdfs_checkpoint_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/models/better_ten/transformer_base
remote_checkpoint_path=${hdfs_checkpoint_path}/${src}_${tgt}
mkdir -p ${local_checkpoint_path}
hadoop fs -mkdir -p ${hdfs_checkpoint_path}
hadoop fs -mkdir -p ${remote_checkpoint_path}

args=(
  ${data_path}
  --save-dir ${local_checkpoint_path}
  --remote-save-dir ${remote_checkpoint_path}
  --task translation
  --source-lang ${src}
  --target-lang ${tgt}
  --dataset-impl "raw"
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1
  --arch transformer
  --optimizer adam
  --adam-betas '(0.9, 0.98)'
  --clip-norm 5
  --lr 5e-4
  --lr-scheduler inverse_sqrt
  --warmup-updates 4000
  --dropout 0.3
  --weight-decay 0.01
  --criterion label_smoothed_cross_entropy
  --label-smoothing 0.1
  --max-tokens 8192
  --update-freq 1
  --max-update 100000
  --fp16
  --valid-subset valid
  --max-sentences-valid 8
  --validate-interval 99999
  --save-interval 99999
  --validate-after-updates 3000
  --validate-interval-updates 3000
  --save-interval-updates 3000
  --keep-interval-updates	10
  --log-interval 10
  --eval-bleu
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.0, "max_len_b": 200}'
  --eval-bleu-detok space
  --eval-bleu-remove-bpe
  --eval-bleu-print-samples
  --best-checkpoint-metric bleu
  --maximize-best-checkpoint-metric
  --activation-fn gelu
  --share-all-embeddings
)

dist-train "${args[@]}"