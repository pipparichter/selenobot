#!/bin/bash

feature_types=("plm_esm_cls" "plm_esm_gap" "plm_pt5" "len" "aa_1mer")

model_types=("2c" "3c_xl" "3c")
data_dir="../data/model_organisms/"
file_names=("afis.h5" "ecol.h5" "paer.h5" "bsub.h5" "mtub.h5")

for feature_type in "${feature_types[@]}"; do
    for model_type in "${model_types[@]}"; do
        for file_name in "${file_names[@]}"; do
            python predict.py --model-name "model_${model_type}_${feature_type}" --feature-type "$feature_type" --input-path "$data_dir$file_name" 
        done
    done
done

for feature_type in "${feature_types[@]}"; do
    for model_types in "${model_types[@]}"; do
        python predict.py --model-name "model_${model_type}_${feature_type}" --feature-type "$feature_type" --input-path ../data/${model_type}_val.h5 
        python predict.py --model-name "model_${model_type}_${feature_type}" --feature-type "$feature_type" --input-path ../data/${model_type}_train.h5 
        python predict.py --model-name "model_${model_type}_${feature_type}" --feature-type "$feature_type" --input-path ../data/${model_type}_test.h5 
    done
done

