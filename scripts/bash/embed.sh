#!/bin/bash

MAILUSER=prichter@caltech.edu
MAILTYPE=END
GRES=gpu:1
PARTITION=gpu
MEM=300GB
TIME=10:00:00

# directory="../data/model_organisms/"
directory="../data/"
feature_types="plm_esm plm_pt5 aa_1mer len"
# file_names=("gtdb_ecol_metadata.csv" "gtdb_bsub_metadata.csv" "gtdb_mtub_metadata.csv")
file_names=("test_metadata.csv" "train_metadata.csv" "val_metadata.csv")

for file_name in "${file_names[@]}"; do
    output_path=$(echo "$directory$file_name" | sed "s/_metadata\.csv/.h5/")
    job_name="embed_$file_name"
    cmd="python embed.py --input-path \"$directory$file_name\" --feature-types \"$feature_types\" --output-path \"$output_path\""
    sbatch --mem="$MEM" --time="$TIME" --partition="$PARTITION" --gres="$GRES" --job-name "$job_name" --mail-user="$MAILUSER" --mail-type="$MAILTYPE" --wrap "$cmd"
done
