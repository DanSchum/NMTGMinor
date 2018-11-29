#!/usr/bin/env bash


source /home/dschumacher/dschumacher_working_dir/anaconda/bin/activate /home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6_cuda_80

python3.6 train.py \
-data /project/student_projects2/dschumacher/preprocessing/after_preprocessing/papers_arxiv_own/category_computer_science/transformer_preproc_2018_11_29/transformer_moses_subword_nmt_2018_11_29.train.pt \
-data_format raw \
-model transformer \
-learning_rate 0.001 \
-layers 6 \
-log_interval 500 \
-save_every 1000 \
-gpus 0 \
-batch_size_words 1024 \
-batch_size_sents 1048576 \
-save_model $1
#-batch_size_update 4096

#srun -p lowGPU -w i13hpc51 nvidia-smi

#srun -c 1 --mem=10000 -t 5-00 -p lowGPU -w i13hpc51 --gres=gpu:1 cmd_train_command_wrapper.sh logs/manualLog.log
