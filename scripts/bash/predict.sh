#!/bin/bash

feature_types = ("plm_esm_cls" "plm_esm_gap" "plm_pt5" "len" "aa_1mer")
n_classes = (2 3)

for feature_type in "${feature_types[@]}"; do
    for n_classes in "${n_classes[@]}"; do
        python predict.py --model-name "model_${n_classes}c_${feature_type}" --feature-type "$feature_type" --input-path ../data/2c_val.h5 
        python predict.py --model-name "model_${n_classes}c_${feature_type}" --feature-type "$feature_type" --input-path ../data/2c_train.h5 
        python predict.py --model-name "model_${n_classes}c_${feature_type}" --feature-type "$feature_type" --input-path ../data/2c_test.h5 
    done
done

