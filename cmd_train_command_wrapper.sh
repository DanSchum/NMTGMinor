#!/usr/bin/env bash

# $1 : Input of *.train.pt file as training input
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

timestampValue=$(timestamp)
outputPath=$2"/Training_output_"$timestampValue
mkdir $outputPath
outputModel=$2"/Training_output_"$timestampValue"/model_"



outputReadme=$outputPath"/README.txt"
touch $outputReadme
echo "Training Input File: "$1 >> $outputReadme
echo "Output Path: "$outputPath >> $outputReadme


source /home/dschumacher/dschumacher_working_dir/anaconda/bin/activate /home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6_cuda_90

python3.6 train.py \
-data $1 \
-data_format raw \
-model transformer \
-learning_rate 0.001 \
-layers 6 \
-log_interval 1000 \
-save_every 2000 \
-gpus 0 \
-batch_size_words 16384 \
-batch_size_sents 262144 \
-max_generator_batches 32 \
-save_model $outputModel
#-batch_size_update 4096

#srun -p lowGPU -w i13hpc51 nvidia-smi

#srun -c 1 --mem=10000 -t 5-00 -p lowGPU -w i13hpc51 --gres=gpu:1 cmd_train_command_wrapper.sh logs/manualLog.log
