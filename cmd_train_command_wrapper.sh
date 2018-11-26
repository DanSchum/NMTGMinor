#!/bin/sh
/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 train.py \
-data /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/category_computer_science/transformer_preproc_2018_11_26/transformer_moses_subword_nmt_2018_11_22.train.pt \
-data_format raw \
-model transformer \
-learning_rate 0.001 \
-layers 6 \
-log_interval 100 \
-save_every 3000 \
-save_model $1
#-batch_size_words 4096 \
#-batch_size_sents 64 \
#-batch_size_update 4096
