#!/bin/sh
/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 train.py \
-data /project/iwslt2014c/MT/user/dschumacher/Masterarbeit_Misc/MiscProjectPycharm/preprocessing_output/prepared_input_data \
-data_format bin \
-model transformer 
