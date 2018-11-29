#!/usr/bin/env bash

#Dont forget / at end of save_data path!!!!!!!!!!!s

/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 preprocess.py \
-train_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/category_computer_science/subword_mnt_2018_11_26/splits/training_source_bpe_source_summaries.txt \
-train_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/category_computer_science/subword_mnt_2018_11_26/splits/training_target_bpe_target_titles.txt \
-valid_src /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/category_computer_science/subword_mnt_2018_11_26/splits/validation_source_bpe_source_summaries.txt \
-valid_tgt /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/category_computer_science/subword_mnt_2018_11_26/splits/validation_target_bpe_target_titles.txt \
-save_data /project/student_projects2/dschumacher/preprocessing/after_preprocessing/papers_arxiv_own/category_computer_science/transformer_preproc_2018_11_29/transformer_moses_subword_nmt_2018_11_29  \
-format raw \
-sort_by_target \
-src_seq_length 1024 \
-tgt_seq_length 512

./start_training.sh