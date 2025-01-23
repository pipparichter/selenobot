#!/bin/bash

MAILUSER=prichter@caltech.edu
MAILTYPE=END
GRES=gpu:1
PARTITION=gpu
MEM=300GB
TIME=100:00:00

directory="../data/model_organisms/"
feature_types=("plm_pt5" "plm_esm")
epochs=200
output_dim=2

# What does the @ do?
# It means "expand all elements of the array" when used in a loop. It iterates over each element in the array

for feature_type in "${feature_types[@]}"; do
    job_name="train_${feature_type}_${output_dim}"
    cmd="python train.py --feature-type \"$feature_type\" --epochs $epochs --output-dim $output_dim"
    sbatch --mem="$MEM" --time="$TIME" --partition="$PARTITION" --job-name "$job_name" --gres="$GRES" --mail-user="$MAILUSER" --mail-type="$MAILTYPE" --wrap "$cmd"
done
