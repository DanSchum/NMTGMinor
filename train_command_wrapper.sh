#!/bin/sh
/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 train.py \
-data /home/dschumacher/dschumacher_working_dir/training_data/ted_talks_english/after_preprocessing/preprocessing_19_10_18_bpe_complete/preproc_output \
-data_format bin \
-model transformer \
-batch_size_words 4096 \
-batch_size_sents 64 \
-batch_size_update 4096
