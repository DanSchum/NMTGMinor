#!/usr/bin/env bash

/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 preprocess.py \
-train_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/preproc_bpe_2018_11_08/splits/training_source_encoded_preproc_source_summaries.txt \
-train_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/preproc_bpe_2018_11_08/splits/training_target_encoded_preproc_target_titles.txt \
-valid_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/preproc_bpe_2018_11_08/splits/validation_source_encoded_preproc_source_summaries.txt \
-valid_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/preproc_bpe_2018_11_08/splits/validation_target_encoded_preproc_target_titles.txt \
-save_data /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/transformer_preproc_2018_11_11 \
-format raw \
-src_seq_length 600 \
-tgt_seq_length 400
