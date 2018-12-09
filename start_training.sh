#!/usr/bin/env bash

# $1 : Input of *.train.pt file as training input (e.g. /home/dschumacher/dschumacher_working_dir/preprocessing/after_preprocessing/cnnDailyMail/transformer_preproc_2018_12_09/transformer_moses_subword_nmt_2018_12_09)
# $2: Output Path, where models should be stored.

# Define a timestamp function
timestamp() {
  date +"%Y_%m_%d_%H_%M_%S"
}

if [ ! -f "$1" ]; then
  echo "Input File (*.train.pt) not found"
  exit 1
fi


if [ ! -d "$2" ]; then
  echo "Output path not found"
  exit 1
fi

# start training
timestampValue=$(timestamp)
outputFilename="logs/outputTraining_"$timestampValue".log"

sbatch -c 1 --mem=12000 -t 5-00 -p lowGPU -w i13hpc51 --gres=gpu:1 -o $outputFilename -e $outputFilename cmd_train_command_wrapper.sh $1 $2
