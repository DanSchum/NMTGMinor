#!/bin/bash

# Define a timestamp function
timestamp() {
  date +"%Y_%m_%d_%H_%M_%S"
}

# start training
timestampValue=$(timestamp)
outputFilename="logs/outputEvaluation_"$timestampValue".log"
#echo $outputFilename
sbatch -c 1 --mem=10000 -t 1-00 -p lowGPU -w i13hpc51 -o $outputFilename -e $outputFilename eval_command_wrapper.sh $1
