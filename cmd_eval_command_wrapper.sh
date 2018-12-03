#!/bin/bash

#Parameter of Script
# $1: (First Parameter) Model which should be used for translation/ evaluation
# $2: (Second Parameter) Validation dataset Source
# $3: (Third Parameter) Validation dataset Target
# $4: (Fourth Parameter) Output Path, where the output should be stored. (!!!No ending / at path!!!)


if [ ! -f "$1" ]; then
  echo "Model not existing"
  exit 1
fi

if [ ! -f "$2" ]; then
  echo "Validation Source dataset not existing"
  exit 1
fi


if [ ! -f "$3" ]; then
  echo "Validation target dataset not existing"
  exit 1
fi


if [ ! -d "$4" ]; then
  echo "Output path not found"
  exit 1
fi


timestamp() {
  date +"%Y_%m_%d_%H_%M_%S"
}

# start training
timestampValue=$(timestamp)
outputDirectory=$4"/eval_"$timestampValue"/"

if [ ! -d "$outputDirectory" ]; then
  mkdir $outputDirectory
fi

echo $1

#Write Hyperparameter to README.txt file
touch README.txt
echo "Model: "$1 > README.txt
echo "Source Dataset for Eval: "$2 > README.txt
echo "Target Dataset for Eval: "$3 > README.txt
echo "Output path of results: "$4 > README.txt


source /home/dschumacher/dschumacher_working_dir/anaconda/bin/activate /home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6_cuda_80

#Parameter of script is model
#Hyperparameters are set like mentioned in Wiki Article Paper: https://arxiv.org/pdf/1801.10198.pdf
python3.6 translate.py \
-model $1 \
-src $2 \
-output $outputDirectory"evalOutput.txt" \
-beam_size 4 \
-gpu 0 \
-replace_unk \
-tgt $3 \
-alpha 0.6 #Length Penalty coefficient (larger alpha results in longer translations)
#-batch_size 1048576 \
