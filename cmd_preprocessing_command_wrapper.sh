#!/usr/bin/env bash

outputPath="/home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/cnnDailyMail/transformer_preproc_2018_12_11/"
nameOutputModel="transformer_moses_subword_nmt_2018_12_10"
sourcePath="/home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/cnnDailyMail/subword_nmt_2018_12_10/splits/"

trainSourceFilename="training_source_cleaned_bpe_articles_copy.txt"
trainTargetFilename="training_target_cleaned_bpe_abstracts_copy.txt"
validSourceFileName="validation_source_cleaned_bpe_articles_copy.txt"
validTargetFileName="validation_target_cleaned_bpe_abstracts_copy.txt"

src_seq_length=3072
tgt_seq_length=1000

logFile=$outputPath"README.txt"
touch $logFile
echo "Source Path: "$sourcePath >> $logFile
echo "Target Path: "$outputPath >> $logFile
echo "Source Sequence Length: "$src_seq_length>> $logFile
echo "Target Sequence Length: "$tgt_seq_length >> $logFile


source /home/dschumacher/dschumacher_working_dir/anaconda/bin/activate /home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6_cuda_90

python3.6 preprocess.py \
-train_src $sourcePath$trainSourceFilename \
-train_tgt $sourcePath$trainTargetFilename \
-valid_src $sourcePath$validSourceFileName \
-valid_tgt $sourcePath$validTargetFileName \
-save_data $outputPath$nameOutputModel \
-format raw \
-sort_by_target \
-src_seq_length $src_seq_length \
-tgt_seq_length $tgt_seq_length


trainingOutput="/home/dschumacher/dschumacher_working_dir_sp2/models/cnnDailyMail/"

#./start_training.sh $outputPath$nameOutputModel".train.pt" $trainingOutput
