#!/bin/bash

src=$1
tgt=$2

export http_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080
export https_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080

python3 setup.py build_ext --inplace
pip install .
pip install sacremoses

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

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU --nnodes=$ARNOLD_NUM  --node_rank=$ARNOLD_ID \
--master_addr=$ARNOLD_WORKER_0_HOST --master_port=$ARNOLD_WORKER_0_PORT fairseq_cli/train.py ${data_path} \
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
