#!/bin/bash

# Define a timestamp function
timestamp() {
  date +"%Y_%m_%d_%H_%M_%S"
}

# start training
timestampValue=$(timestamp)
outputFilename="logs/outputTraining_"$timestampValue".log"
mkdir "/home/dschumacher/dschumacher_working_dir/models/papers_arxiv_own/category_computer_science/Training_output_"$timestampValue
outputModel="/home/dschumacher/dschumacher_working_dir/models/papers_arxiv_own/category_computer_science/Training_output_"$timestampValue"/model_"
#echo $outputFilename
sbatch -c 1 --mem=10000 -t 5-00 -p lowGPU -w i13hpc51 -o $outputFilename -e $outputFilename cmd_train_command_wrapper.sh $outputModel
