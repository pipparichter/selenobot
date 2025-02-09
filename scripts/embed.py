from selenobot.embedders import *
import argparse 
import os
import pandas as pd 
import re

FEATURE_TYPES = ['plm_esm_log', 'plm_esm_cls', 'plm_esm_gap', 'plm_esm_gap_last_10', 'plm_pt5', 'plm_pt5_last_10', 'len', 'aa_1mer']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--feature-types', nargs='+', default=FEATURE_TYPES)
    args = parser.parse_args()
    
    if args.output_path is None:
        output_path = args.input_path.replace('metadata_', '').replace('.csv', '.h5')
    else:
        output_path = args.output_path 

    # NOTE: Will this throw an error if a partial column isn't present?
    df = pd.read_csv(args.input_path, index_col=0, dtype={'partial':str})
    embed(df, path=output_path, feature_types=args.feature_types, overwrite=args.overwrite)


