#!/bin/sh
/project/iwslt2014c/MT/user/dschumacher/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 train.py \
-data /project/iwslt2014c/MT/user/dschumacher/preprocessing/after_preprocessing/papers_arxiv_own/category_computer_science/transformer_preproc_2018_11_26/transformer_moses_subword_nmt_2018_11_26.train.pt \
-data_format raw \
-model transformer \
-learning_rate 0.001 \
-layers 6 \
-log_interval 1 \
-save_every 1 \
-save_model $1
#-batch_size_words 4096 \
#-batch_size_sents 64 \
#-batch_size_update 4096
