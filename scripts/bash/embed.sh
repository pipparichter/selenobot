#!/bin/bash

MAILUSER=prichter@caltech.edu
MAILTYPE=ALL
GRES=gpu:1
PARTITION=gpu
MEM=500GB
TIME=24:00:00

directory="../data/model_organisms/"
file_names=("metadata_ecol.csv" "metadata_bsub.csv" "metadata_mtub.csv" "metadata_afis.csv" "metadata_paer.csv")


# for file_name in "${file_names[@]}"; do
#     job_name="embed_$file_name"
#     cmd="python embed.py --input-path \"$directory$file_name\""
#     sbatch --mem="$MEM" --time="$TIME" --partition="$PARTITION" --gres="$GRES" -o "$job_name.out" --job-name "$job_name" --mail-user="$MAILUSER" --mail-type="$MAILTYPE" --wrap "$cmd"
# done


directory="../data/"
# file_names=($(ls "$directory" | grep '^[0-9]\+c_metadata_.*\.csv$'))
file_names=("2c_metadata_test.csv" "2c_metadata_train.csv" "2c_metadata_val.csv")

for file_name in "${file_names[@]}"; do
    job_name="embed_$file_name"
    cmd="python embed.py --input-path \"$directory$file_name\""
    sbatch --mem="$MEM" --time="$TIME" --partition="$PARTITION" --gres="$GRES" -o "$job_name.out" --job-name "$job_name" --mail-user="$MAILUSER" --mail-type="$MAILTYPE" --wrap "$cmd"
done
