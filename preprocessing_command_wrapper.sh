#!/usr/bin/env bash

/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 preprocess.py \
-train_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/bpe_23_10_18/encoded_preproc_train.article.txt \
-train_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/bpe_23_10_18/encoded_preproc_train.title.txt \
-valid_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/bpe_23_10_18/encoded_preproc_valid.article.filter.txt \
-valid_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/bpe_23_10_18/encoded_preproc_valid.title.filter.txt \
-save_data /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/transformer_preproc_bpe_24_10_18/preproc_output \
-src_seq_length 2048 \
-tgt_seq_length 512
