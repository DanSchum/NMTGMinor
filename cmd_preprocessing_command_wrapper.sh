#!/usr/bin/env bash

/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 preprocess.py \
-train_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/clean_low_tokenization_22_10_18/preproc_train.article.txt \
-train_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/clean_low_tokenization_22_10_18/preproc_train.title.txt \
-valid_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/clean_low_tokenization_22_10_18/preproc_valid.article.filter.txt \
-valid_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/clean_low_tokenization_22_10_18/preproc_valid.title.filter.txt \
-save_data /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/transformer_preproc_no_bpe_2018_11_15  \
-format raw \
-src_seq_length 512 \
-tgt_seq_length 256
