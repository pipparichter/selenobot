#!/bin/bash

MAILUSER=prichter@caltech.edu
MAILTYPE=ALL
GRES=gpu:1
PARTITION=gpu
MEM=600GB
TIME=24:00:00

# directory="../data/model_organisms/"
# file_names=("gtdb_ecol_metadata.csv" "gtdb_bsub_metadata.csv" "gtdb_mtub_metadata.csv" "gtdb_afis_metadata.csv" "gtdb_paer_metadata.csv")


# for file_name in "${file_names[@]}"; do
#     output_path=$(echo "$directory$file_name" | sed "s/_metadata\.csv/.h5/")
#     job_name="embed_$file_name"
#     cmd="python embed.py --input-path \"$directory$file_name\" --output-path \"$output_path\""
#     sbatch --mem="$MEM" --time="$TIME" --partition="$PARTITION" --gres="$GRES" -o "$job_name.out" --job-name "$job_name" --mail-user="$MAILUSER" --mail-type="$MAILTYPE" --wrap "$cmd"
# done


directory="../data/"
# file_names=($(ls "$directory" | grep '^[0-9]\+c_metadata_.*\.csv$'))
file_names=("2c_metadata_test.csv")

for file_name in "${file_names[@]}"; do
    job_name="embed_$file_name"
    cmd="python embed.py --input-path \"$directory$file_name\""
    sbatch --mem="$MEM" --time="$TIME" --partition="$PARTITION" --gres="$GRES" -o "$job_name.out" --job-name "$job_name" --mail-user="$MAILUSER" --mail-type="$MAILTYPE" --wrap "$cmd"
done
