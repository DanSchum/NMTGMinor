#!/bin/bash

# Define a timestamp function
timestamp() {
  date +"%Y_%m_%d_%H_%M_%S"
}

# start training
timestampValue=$(timestamp)
outputFilename="logs/outputPreprocessing_"$timestampValue".log"
#echo $outputFilename
sbatch -c 1 --mem=10000 -t 3-00 -p HPC -o $outputFilename -e $outputFilename cmd_preprocessing_command_wrapper.sh

