#!/usr/bin/env bash

/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 preprocess.py \
-train_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/bpe_25_10_18/encoded_train.article.txt \
-train_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/bpe_25_10_18/encoded_train.title.txt \
-valid_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/bpe_25_10_18/encoded_valid.article.filter.txt \
-valid_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/bpe_25_10_18/encoded_valid.title.filter.txt \
-save_data /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/transformer_preproc_bpe_06_11_18/preproc_output \
-format raw \
-src_seq_length 333 \
-tgt_seq_length 333
