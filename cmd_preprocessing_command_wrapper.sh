#!/usr/bin/env bash

outputPath="/project/student_projects2/dschumacher/preprocessing/after_preprocessing/Gigaword/transformer_2018_11_29/"
nameOutputModel="transformer_moses_subword_nmt_2018_11_29"
sourcePath="/home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/Gigaword/subword_nmt_2018_11_22_8th/"

trainSourceFilename="bpe_train.article.txt"
trainTargetFilename="bpe_train.title.txt"
validSourceFileName="bpe_valid.article.filter.txt"
validTargetFileName="bpe_valid.title.filter.txt"

src_seq_length=512
tgt_seq_length=256

logFile=$outputPath"README.txt"
touch $logFile
echo "Source Path: "$sourcePath >> $logFile
echo "Target Path: "$outputPath >> $logFile
echo "Source Sequence Length: "$src_seq_length>> $logFile
echo "Target Sequence Length: "$tgt_seq_length >> $logFile

/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 preprocess.py \
-train_src $sourcePath$trainSourceFilename \
-train_tgt $sourcePath$trainTargetFilename \
-valid_src $sourcePath$validSourceFileName \
-valid_tgt $sourcePath$validTargetFileName \
-save_data $outputPath$nameOutputModel \
-format raw \
-sort_by_target \
-src_seq_length $src_seq_length \
-tgt_seq_length $tgt_seq_length

./start_training.sh