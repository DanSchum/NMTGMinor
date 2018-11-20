#!/bin/bash


#Parameter of Script
# $1: (First Parameter) Model which should be used for translation/ evaluation
# $2: (Second Parameter) Output Path, where the output should be stored. (!!!No ending / at path!!!)


# Define a timestamp function
timestamp() {
  date +"%Y_%m_%d_%H_%M_%S"
}

# start training
timestampValue=$(timestamp)
outputFilename="logs/outputEvaluation_"$timestampValue".log"
#echo $outputFilename
sbatch -c 1 --mem=10000 -t 1-00 -p lowGPU -w i13hpc51 -o $outputFilename -e $outputFilename cmd_eval_command_wrapper.sh $1 $2
