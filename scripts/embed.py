from selenobot.embedders import *
import argparse 
# from files import FastaFile
import os
import pandas as pd 
import re


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default=None, help='The path where the embeddings will be written.')
    parser.add_argument('--input-path', type=str, default=None, help='The path to the CSV file with the sequences and metadata.')
    parser.add_argument('--append', action='store_true')
    parser.add_argument('--types', nargs='+', default=['plm', 'len', 'aa_1mer'])
    args = parser.parse_args()

    if args.output_path is None:
        input_dir = os.path.dirname(args.input_path) # Get the directory of the input file. 
        input_file_name = os.path.basename(args.input_path) 
        output_path = os.path.join(input_dir, input_file_name.replace('.csv', '.h5'))
    else:
        output_path = args.output_path

    embedders = []
    for type_ in args.types:
        if re.match('aa_([0-9]+)mer', type_) is not None:
            k = int(re.match('aa_([0-9]+)mer', type_).group(1))
            embedders.append(KmerEmbedder(k=k))
        if type_ == 'plm':
            embedders.append(PLMEmbedder())
        if type_ == 'len':
            embedders.append(LengthEmbedder())
    print('Generating embeddings of the following types:', ', '.join([embedder.type for embedder in embedders]), flush=True)

    # NOTE: Will this throw an error if a partial column isn't present?
    df = pd.read_csv(args.input_path, index_col=0, dtype={'partial':str})
    embed(df, path=output_path, embedders=embedders, append=args.append)

    print(f'Embeddings written to {output_path}.', flush=True)


