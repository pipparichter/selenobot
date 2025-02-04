#!/bin/bash

python predict.py --model-name model_2c_plm_esm_cls_add_length_feature --feature-type plm_esm_cls --input-path ../data/2c_val.h5 --add-length-feature
python predict.py --model-name model_2c_plm_esm_cls_add_length_feature --feature-type plm_esm_cls --input-path ../data/2c_test.h5 --add-length-feature

python predict.py --model-name model_2c_plm_esm_log_aa_tokens_only --feature-type plm_esm_log --input-path ../data/2c_val.h5 --aa-tokens-only
python predict.py --model-name model_2c_plm_esm_log_aa_tokens_only --feature-type plm_esm_log --input-path ../data/2c_test.h5 --aa-tokens-only

python predict.py --model-name model_2c_plm_esm_log --feature-type plm_esm_log --input-path ../data/2c_val.h5 
python predict.py --model-name model_2c_plm_esm_log --feature-type plm_esm_log --input-path ../data/2c_test.h5 

python predict.py --model-name model_2c_plm_esm_cls --feature-type plm_esm_cls --input-path ../data/2c_val.h5 
python predict.py --model-name model_2c_plm_esm_cls --feature-type plm_esm_cls --input-path ../data/2c_test.h5 

python predict.py --model-name model_2c_plm_esm_gap --feature-type plm_esm_gap --input-path ../data/2c_val.h5 
python predict.py --model-name model_2c_plm_esm_gap --feature-type plm_esm_gap --input-path ../data/2c_test.h5 

python predict.py --model-name model_2c_plm_pt5 --feature-type plm_pt5 --input-path ../data/2c_val.h5 
python predict.py --model-name model_2c_plm_pt5 --feature-type plm_pt5 --input-path ../data/2c_test.h5

python predict.py --model-name model_2c_len --feature-type len --input-path ../data/2c_val.h5 
python predict.py --model-name model_2c_len --feature-type len --input-path ../data/2c_test.h5

python predict.py --model-name model_2c_aa_1mer --feature-type aa_1mer --input-path ../data/2c_val.h5 
python predict.py --model-name model_2c_aa_1mer --feature-type aa_1mer --input-path ../data/2c_test.h5 