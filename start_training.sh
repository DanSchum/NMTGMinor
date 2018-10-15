#!/bin/bash

# Define a timestamp function
timestamp() {
  date +"%Y_%m_%d_%T"
}

# do something...
timestampValue=$(timestamp)
outputFilename="outputTraining_"+$timestampValue+".log"
sbatch -c 1 --mem=4000 -t 1-00 -p lowGPU -w i13hpc51 -o $outputFilename -e $outputFilename train_command_wrapper.sh
