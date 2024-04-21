'''A script for embedding protein sequences in a FASTA file.'''
import sys

from selenobot.embedders import PlmEmbedder
from selenobot.utils import dataframe_from_fasta, csv_size
import subprocess
from typing import NoReturn
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse

TMP1 = 'tmp.1.csv' # Temporary file for the metadata. 
TMP2 = 'tmp.2.csv' # Temporary file for the embeddings. 

def create_embedding_file(path:str, chunk_size:int=100) -> NoReturn:
    '''Combine the data stored in each temporary file (the embeddings and metadata) in chunks, and store in 
    a new CSV file specified by the path parameter. 
    
    :param path: The path to the final embeddings dataset. .
    :chunk_size: The size of the chunks for processing.
    '''
    embedding_ids = pd.read_csv(TMP2, usecols=['id'])['id'].values.ravel() # Read the IDs in the embedding file to avoid loading the entire thing into memory.
    reader = pd.read_csv(TMP1, index_col=['id'], chunksize=chunk_size) # Use read_csv to load the metadata one chunk at a time. 

    is_first_chunk = True
    n_chunks = csv_size(TMP1) // chunk_size + 1
    for chunk in tqdm(reader, desc='add_embeddings', total=n_chunks):
        # Get the indices of the embedding rows corresponding to the data chunk. Make sure to shift the index up by one to account for the header. 
        idxs = np.where(np.isin(embedding_ids, chunk.index, assume_unique=True))[0] + 1 
        idxs = [0] + list(idxs) # Add the header index so the column names are included. 
        # Read in the embedding rows, skipping rows which do not match a gene ID in the chunk. 
        chunk = chunk.merge(pd.read_csv(TMP2, skiprows=lambda i : i not in idxs), on='id', how='inner')
        # Check to make sure the merge worked as expected. Subtract 1 from len(idxs) to account for the header row.
        assert len(chunk) == (len(idxs) - 1), f'Data was lost while merging embedding data.'
        
        chunk.to_csv(path, header=is_first_chunk, mode='w' if is_first_chunk else 'a') # Only write the header for the first file. 
        is_first_chunk = False


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='The path to the FASTA file containing the sequences to embed.', type=str)
    parser.add_argument('output', help='The path write the embeddings to.', type=str)
    parser.add_argument('-m', '--model', help='The name of the protein language model to use for generating embeddings.', default='Rostlab/prot_t5_xl_half_uniref50-enc', type=str)

    args = parser.parse_args()

    # Make sure the file types are correct. 
    input_file_type = args.input.split('.')[-1]
    output_file_type = args.output.split('.')[-1] 
    assert input_file_type in ['fasta', 'faa', 'fa'], 'Unsupported input file type. Must be in FASTA format.'
    assert output_file_type == 'csv', 'Unsupported input file type. Must be a CSV file.'

    # Instantiate the PLM embedder with the model name. 
    embedder = PlmEmbedder(args.model)

    df = dataframe_from_fasta(args.input, parse_header=False) # Load the FASTA file containing the sequences to embed. 
    df.set_index('id').to_csv(TMP1) # Write the DataFrame to a temporary CSV file. 

    seqs = list(df['seq'].values) # Get the amino acid sequences in the file as a list of strings. 
    ids = list(df['id'].values)

    # Write the embeddings and the corresponding IDs to a different CSV output file. This file is temporary.
    print('Generating PLM embeddings...') 
    embeddings, ids = embedder(seqs, ids) # Note that IDs are redefined here, as the embedding process scrambles the order.
    print('Done.')
    print('Writing embeddings to temporary file...')
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df['id'] = ids # Set an ID column so the embeddings can be matched to the metadata. 
    embeddings_df.set_index('id').to_csv(TMP2)
    print('Done')

    # Combine the embeddings (in TMP2) with the metadata in TMP1 piece-by-piece (to avoid memory issues)
    print('Combining metadata and PLM embeddings...')
    create_embedding_file(args.output)
    print('Done.')
    print(f'Embedding written to {args.output}.')

    # Remove the temporary files.
    subprocess.run(f'rm {TMP1}', shell=True, check=True)
    subprocess.run(f'rm {TMP2}', shell=True, check=True)


