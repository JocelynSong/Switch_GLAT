#!/bin/bash

pair=$1  # input language pair

# data paths
MAIN_PATH=.
DATA_PATH=$MAIN_PATH/data
PREPROCESS=$MAIN_PATH/fairseq_cli/process_raw_data_to_path.py
REMOTE_PATH=hdfs://haruna/home/byte_arnold_hl_mlnlc/user/songzhenqiao/multilingual_glat/total_data/wmt_en_de_fr_ro_ru_zh/new_better_diffusion_data

for rate in 0.2 0.3 0.4 0.5; do
for ((rank=0;rank<8;rank++)); do
PROCESSED_PATH=$DATA_PATH/rate${rate}/rank${rank}

for split in src tgt; do
      python3 $PREPROCESS $PROCESSED_PATH/diffusion.$pair.$split.txt $PROCESSED_PATH/diffusion.$pair.$split.pth

      hadoop fs -put $PROCESSED_PATH/diffusion.$pair.$split.pth $REMOTE_PATH/rate${rate}/rank${rank}
done
done
done

