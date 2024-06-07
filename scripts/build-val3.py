import pandas as pd 
from selenobot.utils import DATA_DIR, SCRIPTS_DIR, dataframe_from_fasta, dataframe_to_fasta
from selenobot.extend import extend, translate
import os
import numpy as np
import subprocess
import time
from tqdm import tqdm 
import matplotlib.pyplot as plt
import warnings 
from selenobot.extend import extend
import sklearn
from typing import List, NoReturn
import wget
import itertools
import subprocess
import scipy
import argparse
import gzip

# TODO: Maybe switch over to using the wget Python package instead of subprocess.run. 
# TODO: It seems like the embed.py file is not correctly setting the index to the id column, and I am not sure why.
# TODO: Add docstrings to functions in this file.  

warnings.filterwarnings('ignore')

# Set the main data directory to be the v3 validation directory. 
VAL_DIR = os.path.join(DATA_DIR, 'val3') 


def download_aa_seqs() -> NoReturn:
    ''''''
    path = os.path.join(VAL_DIR, 'aaseqs.fa') 
    gz_path = os.path.join(VAL_DIR, 'aaseqs.gz') 

    if not os.path.exists(gz_path):
        print(f'download_aa_seqs: Downloading amino acid sequences for all organisms to {gz_path}.')
        subprocess.run(f'wget --quiet https://figshare.com/ndownloader/files/44580544 -O {gz_path}', shell=True, check=True)
    
    if not os.path.exists(path):
        print(f'download_aa_seqs: Extracting compressed amino acid sequences for all organisms to {path}.')
        with gzip.open(gz_path, 'rt') as f:
            text = f.read()
        with open(path, 'w') as f:
            f.write(text) # Write the text to a new file. 


def download_metadata(organism:str) -> NoReturn:
    ''''''
    path = os.path.join(VAL_DIR, organism, 'metadata.csv')

    if not os.path.exists(path):
        print(f'download_metadata: Downloading metadata for organism {organism} to {path}.')
        # tmp_path = os.path.join(VAL_DIR, 'metadata', f'{organism}.tsv') # Write the metadata to a temporary path. 
        subprocess.run(f'wget "https://fit.genomics.lbl.gov/cgi-bin/orgGenes.cgi?orgId={organism}" -O {path} --quiet', shell=True, check=True)

        # Clean up the metadata a bit, and put in CSV format. 
        metadata_df = pd.read_csv(path, delimiter='\t') # Load the gene metadata.
        metadata_df = metadata_df.rename(columns={'locusId':'id', 'scaffoldId':'scaffold_id'}).drop(columns=['sysName'])
        metadata_df = metadata_df.set_index('id')
        metadata_df.to_csv(path) # Overwrite the original file. 


def download_fitness(organism:str) -> pd.DataFrame:
    ''''''
    path = os.path.join(VAL_DIR, organism, 'fitness.csv')

    if not os.path.exists(path):
        print(f'download_fitness: Downloading fitness data for organism {organism} to {path}.')
        subprocess.run(f'wget --quiet "https://fit.genomics.lbl.gov/cgi-bin/createFitData.cgi?orgId={organism}" -O {path}', shell=True, check=True)

        # Clean up the data a bit, and put in CSV format (it was downloaded as tab-separated)
        fitness_df = pd.read_csv(path, delimiter='\t') # Load the gene metadata.
        fitness_df = fitness_df.rename(columns={'locusId':'id', 'scaffold':'scaffold_id'})
        fitness_df = fitness_df.drop(columns=['geneName', 'desc', 'sysName', 'orgId']) # Drop some columns which seem to be irrelevant or unimportant. 
        fitness_df = fitness_df.set_index('id')
        fitness_df.to_csv(path)


def download_genome(organism:str) -> pd.DataFrame:
    ''''''
    path = os.path.join(VAL_DIR, organism, 'genome.fna') # Add the filename to the path.

    if not os.path.exists(path):
        print(f'download_genome: Downloading genome for organism {organism} to {path}.')
        subprocess.run(f'wget --quiet "https://fit.genomics.lbl.gov/cgi-bin/orgSeqs.cgi?orgId={organism}&type=nt" -O {path}', shell=True, check=True)


def predict(organism:str) -> pd.DataFrame:
    ''''''
    # This file contains predicted genes for every organism. Need to filter for genes which only belong to the organism. 
    aa_seqs_df = dataframe_from_fasta(os.path.join(VAL_DIR, 'aaseqs.fa'), parser=None)
    aa_seqs_df = aa_seqs_df[aa_seqs_df['id'].str.contains(organism)] # Get all sequences for the specified organism.
    aa_seqs_df['id'] = aa_seqs_df['id'].str.replace(f'{organism}:', '') # Remove the organism name from the locus ID. 

    pred_path = os.path.join(VAL_DIR, organism, 'predictions.csv')
    fasta_path = os.path.join(VAL_DIR, organism, 'genes.fa')
    emb_path = os.path.join(VAL_DIR, organism, 'embeddings.csv') 

    # First, we need to generate a FASTA file containing only the genes from the specified organism. 
    dataframe_to_fasta(aa_seqs_df, fasta_path) # Convert to a FASTA file so it can be used with the scripts. 

    # Embed the genes in the DataFrame and make predictions using a pre-trained model. 
    if not os.path.exists(emb_path):
        subprocess.run(f"python {os.path.join(SCRIPTS_DIR, 'embed.py')} {fasta_path} {emb_path}", shell=True, check=True)
    if not os.path.exists(pred_path):
        subprocess.run(f"python {os.path.join(SCRIPTS_DIR, 'predict.py')} {emb_path} {pred_path}", shell=True, check=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--organism', type=str, default='acidovorax_3H11') 
    args = parser.parse_args()

    organism = args.organism
    organism_path = os.path.join(VAL_DIR, organism) # Path to the directory where all the data for the organism will be stored.
    # If the path for the organism data does not yet exist, go ahead and make it. 
    if not os.path.exists(organism_path):
        os.mkdir(os.path.join(organism_path))

    download_aa_seqs()
    download_metadata(organism)
    download_fitness(organism)
    download_genome(organism)

    predict(organism)