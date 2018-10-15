#!/bin/bash


timestamp() {
  date +"%Y_%m_%d_%H_%M_%S"
}

# start training
timestampValue=$(timestamp)
outputDirectory="/home/dschumacher/dschumacher_working_dir/evaluation/evaluation_output/eval_"$timestampValue"/"


if [ ! -d "$outputDirectory" ]; then
  mkdir $outputDirectory
fi

echo $1

#Parameter of script is model
/home/dschumacher/dschumacher_working_dir/anaconda/envs/NMTGMinor_env_python3_6/bin/python3.6 translate.py \
-model $1 \
-src /home/dschumacher/dschumacher_working_dir/evaluation/evaluation_data/validationSplitSource.txt \
-output $outputDirectory"pred.txt"
