#!/bin/bash

export http_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080
export https_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080

python3 setup.py build_ext --inplace
pip install .
pip install sacremoses

# Download and prepare the data
data_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data
local_dataset_path=${data_path}/wmt_en_de_fr_ro_ru_zh

local_root=.
output_path=${local_root}/output
mkdir -p ${output_path}
local_checkpoint_path=${output_path}/save_model
hdfs_checkpoint_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/models/multilingual_transformer
remote_checkpoint_path=${hdfs_checkpoint_path}/wmt_ten
mkdir -p ${local_checkpoint_path}
hadoop fs -mkdir -p ${hdfs_checkpoint_path}
hadoop fs -mkdir -p ${remote_checkpoint_path}

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU --nnodes=$ARNOLD_NUM  --node_rank=$ARNOLD_ID \
--master_addr=$ARNOLD_WORKER_0_HOST --master_port=$ARNOLD_WORKER_0_PORT fairseq_cli/train.py ${local_dataset_path} \
--ddp-backend legacy_ddp \
--save-dir ${local_checkpoint_path} \
--remote-save-dir ${remote_checkpoint_path} \
--task multilingual_translation \
--lang-pairs "de-en,en-de,en-fr,fr-en,en-zh,zh-en,en-ru,ru-en,en-ro,ro-en" \
--dataset-impl "raw" \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--arch multilingual_transformer \
--optimizer adam \
--adam-betas '(0.9, 0.999)' --adam-eps 1e-6 \
--clip-norm 2 \
--lr 5e-4 \
--lr-scheduler inverse_sqrt \
--stop-min-lr 1e-9 \
--warmup-updates 4000 \
--warmup-init-lr 1e-7 \
--dropout 0.1 \
--weight-decay 0.01 \
--max-tokens 8192 \
--update-freq 1 \
--max-update 1000000 \
--max-epoch 200 \
--fp16 \
--valid-subset valid \
--max-sentences-valid 8 \
--validate-interval 99999 \
--save-interval 99999 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--keep-interval-updates	10 \
--log-interval 10 \
--activation-fn gelu \
--share-all-embeddings
