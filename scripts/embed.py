from selenobot.embedders import embed 
import argparse 
# from files import FastaFile
import os
import pandas as pd 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default=None, help='The path where the embeddings will be written.')
    parser.add_argument('--input-path', type=str, default=None, help='The path to the CSV file with the sequences and metadata.')
    args = parser.parse_args()

    if args.output_path is None:
        input_dir = os.path.dirname(args.input_path) # Get the directory of the input file. 
        input_file_name = os.path.basename(args.input_path) 
        output_path = os.path.join(input_dir, input_file_name.replace('.csv', '.h5'))
    else:
        output_path = args.output_path

    # NOTE: Will this throw an error if a partial column isn't present?
    df = pd.read_csv(args.input_path, index_col=0, dtype={'partial':str})
    embed(df, path=output_path)
    print(f'Embeddings written to {output_path}.')


