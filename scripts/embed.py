from selenobot.embedders import *
import argparse 
import os
import pandas as pd 
import re


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--feature-types', nargs='+', default=['plm_esm', 'plm_pt5','len', 'aa_1mer'])
    args = parser.parse_args()

    embedders = []
    for feature_type in args.feature_types:
        if re.match('aa_([0-9]+)mer', feature_type) is not None:
            k = int(re.match('aa_([0-9]+)mer', feature_type).group(1))
            embedders.append(KmerEmbedder(k=k))
        if re.match('plm_([a-z]+)', feature_type) is not None:
            model_name = re.match('plm_([a-z]+)', feature_type).group(1) 
            embedders.append(PLMEmbedder(model_name=model_name))
        if feature_type == 'len':
            embedders.append(LengthEmbedder())

    # NOTE: Will this throw an error if a partial column isn't present?
    df = pd.read_csv(args.input_path, index_col=0, dtype={'partial':str})
    embed(df
    , path=args.output_path, embedders=embedders, overwrite=args.overwrite)


