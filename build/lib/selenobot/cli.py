from selenobot.classifiers import Classifier
from selenobot.utils import WEIGHTS_DIR, DATA_DIR
from selenobot.files import FastaFile, ProteinsFile
from selenobot.datasets import Dataset
import os
import numpy as np
import os
import argparse
import torch
import pandas as pd
import sys, re, os, time, wget
from typing import NoReturn, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import subprocess


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to the input embeddings on which to run the classifier.')
    parser.add_argument('output', help='Path to the file where model predictions will be written.')
    parser.add_argument('--weights', type=str, default=os.path.join(WEIGHTS_DIR, 'plm_model_weights.pth'), help='The path to the stored model weights.')
    args = parser.parse_args()

    model = Classifier(input_dim=1024, hidden_dim=512)
    model.load_state_dict(torch.load(args.weights))
    
    dataset = Dataset(pd.read_csv(args.input)) # Instantiate a Dataset object with the embeddings. 
    reporter = model.predict(dataset)
    predictions = reporter.apply_threshold()

    df = pd.DataFrame({'id':dataset.ids, 'model_output':reporter.outputs, 'prediction':predictions})
    df['seq'] = dataset.seqs # Add sequences to the DataFrame. 
    df.set_index('id').to_csv(args.output)


def embed():
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

    df = dataframe_from_fasta(args.input) # Load the FASTA file containing the sequences to embed. 
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

