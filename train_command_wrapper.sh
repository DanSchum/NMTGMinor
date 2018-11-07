#!/bin/sh
/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 train.py \
-data /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/transformer_preproc_bpe_06_11_18/preproc_output \
-data_format raw \
-model transformer \
-learning_rate 0.001 \
-log_interval 1
#-batch_size_words 4096 \
#-batch_size_sents 64 \
#-batch_size_update 4096
