#!/bin/bash

export http_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080
export https_proxy=http://lab_mt_caojun.sh:N92TPbyQoQfYkROb@10.124.155.170:8080

python3 setup.py build_ext --inplace
pip install .
pip install sacremoses

# AT Data Pth
data_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data/wmt_en_de_fr_ro_ru_zh
local_dataset_path=${data_path}/at_data

# Diffusion data path
diffusion_data_path=${data_path}/nat_diffusion_data

# lcoal model path
local_root=.
output_path=${local_root}/output
mkdir -p ${output_path}
local_checkpoint_path=${output_path}/save_model

# remote model saving path
hdfs_checkpoint_path=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/models/multilingual_nat
remote_checkpoint_path=${hdfs_checkpoint_path}/nat_diffusion_ten_lang
mkdir -p ${local_checkpoint_path}
hadoop fs -mkdir -p ${hdfs_checkpoint_path}
hadoop fs -mkdir -p ${remote_checkpoint_path}

# initialize model path
initialized_model_path=${hdfs_checkpoint_path}/vanilla_MNAT
model_name=checkpoint_zh-en_best.pt
hadoop fs -get ${initialized_model_path}/${model_name} ${local_checkpoint_path}

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
python3 -m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU --nnodes=$ARNOLD_NUM  --node_rank=$ARNOLD_ID \
--master_addr=$ARNOLD_WORKER_0_HOST --master_port=$ARNOLD_WORKER_0_PORT fairseq_cli/train_mglat_diffusion.py ${local_dataset_path} \
--save-dir ${local_checkpoint_path} \
--remote-save-dir ${remote_checkpoint_path} \
--restore-file ${local_checkpoint_path}/${model_name} \
--task multilingual_nat_translation \
--lgs "de-en-fr-ro-ru-zh" \
--mt-steps "de-en,en-de,en-fr,fr-en,en-ro,ro-en,en-ru,ru-en,en-zh,zh-en" \
--metric-pair "de-en" \
--total-sample-updates 600000 \
--minus-p 0.3 \
--dataset-impl "raw" \
--criterion glat_loss --label-smoothing 0.1 \
--arch MGLAT_base \
--noise full_mask \
--optimizer adam \
--adam-betas '(0.9, 0.999)' --adam-eps 1e-6 \
--clip-norm 2 \
--lr 5e-4 \
--lr-scheduler inverse_sqrt \
--stop-min-lr 1e-9 \
--warmup-updates 4000 \
--warmup-init-lr 1e-7 \
--dropout 0.3 \
--annealing-total-num 1200000 \
--weight-decay 0.01 \
--max-tokens 8192 \
--update-freq 1 \
--max-update 6000000 \
--max-epoch 5000 \
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
--eval-bleu \
--eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 5}' \
--eval-bleu-detok space \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--activation-fn gelu \
--share-all-embeddings \
--vanilla-model-bleu '{"de-en": 23.48, "en-de": 16.06, "ro-en": 31.95, "en-ro": 26.63, "ru-en": 24.37, "en-ru": 16.28, "en-zh": 7.49, "zh-en": 9.25, "en-fr": 24.03, "fr-en": 27.1}' \
--diffusion-data-path ${diffusion_data_path} \
--diffusion-generation-interval 10 \
--diffusion-interval 5 \
--diffusion-num 300000 \
--diffusion-steps "de-en-zh,en-de-fr,en-fr-ro,en-ru-zh,en-zh-de,fr-en-de,ro-en-fr,ru-en-ro,zh-en-ru" \
--diffusion-percentage 0.4 \
--diffusion-max-sentence 8 \
--diffusion-length-beam 1 \
--enable-lazy-loader \
--buffer-size 500000 \
--lazy-load-interval 30
