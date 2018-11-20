#!/bin/bash

#Parameter of Script
# $1: (First Parameter) Model which should be used for translation/ evaluation
# $2: (Second Parameter) Output Path, where the output should be stored. (!!!No ending / at path!!!)

timestamp() {
  date +"%Y_%m_%d_%H_%M_%S"
}

# start training
timestampValue=$(timestamp)
outputDirectory=$2"/eval_"$timestampValue"/"


if [ ! -d "$outputDirectory" ]; then
  mkdir $outputDirectory
fi

echo $1

#Parameter of script is model
#Hyperparameters are set like mentioned in Wiki Article Paper: https://arxiv.org/pdf/1801.10198.pdf
/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 translate.py \
-model $1 \
-src /home/dschumacher/dschumacher_working_dir/evaluation/evaluation_data/tedTalks/preproc_bpe_short/encoded_validationSplitSource_lowercase_tokenizer_cleaned_short.txt \
-output $outputDirectory"pred.txt" \
-beam_size 4 \
-alpha 0.6 #Length Penalty coefficient (larger alpha results in longer translations)

