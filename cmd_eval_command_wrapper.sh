#!/bin/bash

#Parameter of Script
# $1: (First Parameter) Model which should be used for translation/ evaluation
# $2: (Second Parameter) Validation dataset
# $3: (Third Parameter) Output Path, where the output should be stored. (!!!No ending / at path!!!)


if [ ! -f "$1" ]; then
  echo "Model not existing"
  exit 1
fi

if [ ! -f "$2" ]; then
  echo "Validation dataset not existing"
  exit 1
fi


if [ ! -d "$3" ]; then
  echo "Output path not found"
  exit 1
fi


timestamp() {
  date +"%Y_%m_%d_%H_%M_%S"
}

# start training
timestampValue=$(timestamp)
outputDirectory=$3"/eval_"$timestampValue"/"


if [ ! -d "$outputDirectory" ]; then
  mkdir $outputDirectory
fi

echo $1

#Parameter of script is model
#Hyperparameters are set like mentioned in Wiki Article Paper: https://arxiv.org/pdf/1801.10198.pdf
/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 translate.py \
-model $1 \
-src $2 \
-output $outputDirectory"evalOutput.txt" \
-beam_size 4 \
-alpha 0.6 #Length Penalty coefficient (larger alpha results in longer translations)

