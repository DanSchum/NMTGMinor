#!/usr/bin/env bash

outputPath="/home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/category_physics/transformer_2018_12_03/"
nameOutputModel="transformer_moses_subword_nmt_2018_12_03"
sourcePath="/home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/papers_arxiv_own/category_physics/subword_nmt_2018_11_27/splits/"

trainSourceFilename="training_source_bpe_source_summaries.txt"
trainTargetFilename="training_target_bpe_target_titles.txt"
validSourceFileName="validation_source_bpe_source_summaries.txt"
validTargetFileName="validation_target_bpe_target_titles.txt"

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


trainingOutput="/home/dschumacher/dschumacher_working_dir_sp2/models/papers_arxiv_own/category_physics/"

./start_training.sh $outputPath$nameOutputModel".train.pt" $trainingOutput