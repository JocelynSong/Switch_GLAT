#!/usr/bin/env bash

cd /opt/tiger/better_mglat

export http_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080
export https_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080


python3 setup.py build_ext --inplace
pip install .
pip install sacremoses

# Download and prepare the data
data_path=hdfs://
save_hdfs_path=hdfs://

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU --nnodes=$ARNOLD_NUM  --node_rank=$ARNOLD_ID \
--master_addr=$ARNOLD_WORKER_0_HOST --master_port=$ARNOLD_WORKER_0_PORT fairseq_cli/train.py $data_path \
--save-dir ${save_hdfs_path} \
--task translation_lev_modified \
--criterion glat_loss \
--arch GLAT_base \
--noise full_mask \
--optimizer adam --adam-betas '(0.9,0.999)' \
--lr 5e-4 \
--lr-scheduler inverse_sqrt \
--stop-min-lr '1e-09' \
--warmup-updates 10000 \
--warmup-init-lr '1e-07' \
--label-smoothing 0.1 \
--dropout 0.3 \
--clip-norm 2 \
--weight-decay 0.01 \
--decoder-learned-pos \
--encoder-learned-pos \
--log-format 'simple' \
--log-interval 50 \
--fixed-validation-seed 7 \
--max-tokens 8192 \
--update-freq 1 \
--max-update 300000 \
--max-source-positions 256 \
--max-target-positions 256 \
--fp16 \
--valid-subset valid \
--max-tokens-valid 80 \
--validate-interval 99999 \
--save-interval 99999 \
--validate-after-updates 3000 \
--validate-interval-updates 3000 \
--save-interval-updates 3000 \
--keep-interval-updates	30 \
--eval-bleu \
--eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 5}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--activation-fn gelu \
--share-all-embeddings
