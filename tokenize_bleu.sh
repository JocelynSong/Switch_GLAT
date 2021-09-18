#!/bin/bash

src=$1
tgt=$2
pair=$src-$tgt

moses_path=mosesdecoder/scripts
tokenizer_path=$moses_path/tokenizer/tokenizer.perl
output_path=./generation/wmt_ten/${src}_${tgt}

perl $tokenizer_path -l $tgt -a < $output_path/$pair.ref.txt > $output_path/$pair.ref.tok.txt
perl $tokenizer_path -l $tgt -a < $output_path/$pair.hypo.txt > $output_path/$pair.hypo.tok.txt
sacrebleu --tok 'none' $output_path/$pair.ref.tok.txt < $output_path/$pair.hypo.tok.txt
