#!/bin/bash

feature_types=("plm_esm_cls" "plm_esm_gap" "plm_pt5" "len" "aa_1mer")
n_classes=("2" "3")
data_dir="../data/model_organisms/"
file_names=("afis.h5" "ecol.h5" "paer.h5" "bsub.h5" "mtub.h5")

# for feature_type in "${feature_types[@]}"; do
#     for n in "${n_classes[@]}"; do
#         for file_name in "${file_names[@]}"; do
#             python predict.py --model-name "model_${n}c_${feature_type}" --feature-type "$feature_type" --input-path "$data_dir$file_name" 
#         done
#     done
# done



for feature_type in "${feature_types[@]}"; do
    for n in "${n_classes[@]}"; do
        python predict.py --model-name "model_${n}c_${feature_type}" --feature-type "$feature_type" --input-path ../data/${n}c_val.h5 
        python predict.py --model-name "model_${n}c_${feature_type}" --feature-type "$feature_type" --input-path ../data/${n}c_train.h5 
        python predict.py --model-name "model_${n}c_${feature_type}" --feature-type "$feature_type" --input-path ../data/${n}c_test.h5 
    done
done

