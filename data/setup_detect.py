'''Code for setting up the detect subdirectory.'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import random
import subprocess
import os
from typing import NoReturn, Dict, List
import sys
import torch
import time
from selenobot.utils import *

# Load information from the configuration file. 
UNIPROT_DATA_DIR = load_config_paths()['uniprot_data_dir']
DETECT_DATA_DIR = load_config_paths()['detect_data_dir']

def setup_train_test_val(
    all_data_path:str=None,
    all_embeddings_path:str=None,
    train_path:str=None,
    test_path:str=None,
    val_path:str=None,
    train_size:int=None,
    test_size:int=None,
    val_size:int=None,
    use_existing_clstr_file:bool=True,
    clstr_file_path:str=os.path.join(UNIPROT_DATA_DIR, 'all_data.clstr')) -> NoReturn:
    '''Splits the data stored in the input path into a train set, test set, and validation set. These sets are disjoint.
    
    args:
        - all_data_path: A path to a FASTA file containing all the UniProt data used for training, testing, etc. the model.
        - all_embeddings_path: A path to a CSV file containing the embeddings of each gene in all_data_path, as well as gene IDs.
        - train_path, test_path, val_path: Paths to the training, test, and validation datasets. 
        - train_size, test_size, val_size: Sizes of the training, test, and validation datasets. 
    '''
    f = 'setup.setup_train_test_val'
    assert (train_size > test_size) and (test_size >= val_size), f'{f}: Expected size order is train_size > test_size >= val_size.'

    # Read in the data, and convert sizes to integers. The index column is gene IDs.
    all_data = pd_from_fasta(all_data_path, set_index=False)

    # This takes too much memory. Will need to do this in chunks. 
    # all_data = all_data.merge(pd.read_csv(all_embeddings_path), on='id')
    # print(f'{f}: Successfully added embeddings to the dataset.') 
    
    # Run CD-HIT on the data stored at the given path, or load an existing clstr_file. .
    clstr_data = pd_from_clstr(clstr_file_path) if use_existing_clstr_file else run_cd_hit(all_data_path, l=MIN_SEQ_LENGTH - 1, n=5) 
    clstr_data_size = len(clstr_data)

    # TODO: Switch over to indices rather than columns, which is faster. 
    # Add the cluster information to the data. 
    all_data = all_data.merge(cluster_data, on='id')
    print(f'{f}: Successfully added homology cluster information to the dataset.') 

    all_data, train_data = sample_homology(all_data, size=len(all_data) - train_size)
    val_data, test_data = sample_homology(all_data, size=val_size)
    
    assert len(train_data) + len(val_data) + len(test_data) == len(clstr_data), f'{f}: Expected {clstr_data_size} sequences present after partitioning, but got {len(train_data) + len(val_data) + len(test_data)}.'

    for data, path in zip([train_data, test_data, val_data], [train_path, test_path, val_path]):
        # Add labels to the DataFrame based on whether or not the gene_id contains a bracket.
        data['label'] = [1 if '[' in gene_id else 0 for gene_id in data.index]
        data.to_csv(path, index=False)
        print(f'{f}: Data successfully written to {path}.')
        add_embeddings_to_file(path, all_embeddings_path)


def add_embeddings_to_file(path:str, embeddings_path:str, chunk_size:int=1000):
    '''Adding embedding information to a CSV file.'''
    f = 'setup.add_embeddings_to_file'

    embeddings_ids = csv_ids(embeddings_path)
    reader = pd.read_csv(path, index_col=['id'], chunksize=chunk_size)
    tmp_file_path = os.path.join(os.path.dirname(path), 'tmp.csv')

    is_first_chunk = True
    n_chunks = csv_size(path) // chunk_size + 1
    for chunk in tqdm(reader, desc=f'{f}: adding embeddings to {path}.', total=n_chunks):
        # Get the indices of the embedding rows corresponding to the data chunk.
        idxs = np.where(np.isin(embeddings_ids, chunk.index, assume_unique=True))[0] + 1 # Make sure to shift the index up by one to include the header. 
        idxs = [0] + list(idxs) # Add the header index for merging. 

        chunk = chunk.merge(pd.read_csv(embeddings_path, skiprows=lambda i : i not in idxs), on='id', how='inner')

        # Subtract 1 from len(idxs) to account for the header row.
        assert len(chunk) == (len(idxs) - 1), f'{f}: Data was lost while merging embedding data.'
        
        chunk.to_csv(tmp_file_path, header=is_first_chunk, mode='w' if is_first_chunk else 'a') # Only write the header for the first file. 
        is_first_chunk = False

    # Replace the old file with tmp. 
    subprocess.run(f'rm {path}', shell=True, check=True)
    subprocess.run(f'mv {tmp_file_path} {path}', shell=True, check=True)


def sample_homology(data:pd.DataFrame, size:int=None):
    '''Subsample the cluster data such that the entirety of any homology group is contained in the sample. 
  
    args:
        - cluster_data: A pandas DataFrame mapping the gene ID to cluster number. 
        - size: The size of the first group, which must also be the smaller group. 
    '''
    f = 'setup.sample_homology'
    assert (len(data) - size) >= size, f'{f}: The size argument must specify the smaller partition. Provided arguments are len(clusters)={len(data)} and size={size}.'

    groups = {'sample':[], 'remainder':[]} # First group is smaller.
    curr_size = 0 # Keep track of how big the sample is, without concatenating DataFrames just yet. 

    ordered_clusters = data.groupby('cluster').size().sort_values(ascending=False).index # Sort the clusters in descending order of size. 

    add_to = 'sample'
    for cluster in tqdm(ordered_clusters, desc=f):
        # If we check for the larger size, it's basically the same as adding to each group in an alternating way, so we need to check for smaller size first.
        cluster = data[data.cluster == cluster]
        if add_to == 'sample' and curr_size < size:
            groups['sample'].append(cluster)
            curr_size += len(cluster)
            add_to = 'remainder'
        else:
            groups['remainder'].append(cluster)
            add_to = 'sample'

    sample, remainder = pd.concat(groups['sample']), pd.concat(groups['remainder'])
    assert len(sample) + len(remainder) == len(data), f"{f}: The combined sizes of the partitions do not add up to the size of the original data."
    assert len(sample) < len(remainder), f'{f}: The sample DataFrame should be smaller than the remainder DataFrame.'

    print(f'{f}: Collected homology-controlled sample of size {len(sample)} ({np.round(len(sample)/len(data), 2) * 100} percent of the input dataset).')
    return sample, remainder


def run_cd_hit(fasta_file_path:str, c:float=0.8, l:int=1, n:int=2) -> pd.DataFrame:
    '''Run CD-HIT on the FASTA file stored in the input path, generating the homology-based sequence similarity clusters.
    
    args:
        - fasta_file_path: Path to the FASTA file on which to run the CD-HIT program. 
        - c: The similarity cutoff for sorting sequences into clusters (see CD-HIT documentation for more information). 
        - l: Minimum sequence length allowed by the clustering algorithm. Note that some truncated sequences
            are filtered out by this parameter (see CD-HIT documentation for more information). 
        - n: Word length (see CD-HIT documentation for more information).
    '''
    # assert (min([len(seq) for seq in fasta_seqs(path)])) > l, 'Minimum sequence length {l + 2} is longer than the shortest sequence.'
    directory, filename = os.path.split(fasta_file_path)
    filename = filename.split('.')[0] # Remove the extension from the filename. 
    subprocess.run(f"{CD_HIT} -i {fasta_file_path} -o {os.path.join(directory, filename)} -c {c} -l {l} -n {n}", shell=True, check=True) # Run CD-HIT.
    subprocess.run(f'rm {os.path.join(directory, filename)}', shell=True, check=True) # Remove the output file with the cluster reps. 

    # Load the clstr data into a DataFrame and return. 
    return pd_from_clstr(os.path.join(directory, filename + '.clstr'))


def setup_detect():
    '''Creates a dataset for the detection classification task, which involves determining
    whether a protein, truncated or not, is a selenoprotein. The data consists of truncated and
    non-truncated selenoproteins, as well as a number of normal full-length proteins equal to all 
    selenoproteins.'''
    
    all_data_size = fasta_size(os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'))

    train_size = int(0.8 * all_data_size)
    test_size = int(0.6 * (all_data_size - train_size)) # Making sure test_size is larger than val_size.
    val_size = all_data_size - train_size - test_size
    sizes = {'train_size':train_size, 'test_size':test_size, 'val_size':val_size}

    # train_data, test_data, and val_data should have sequence information. 
    setup_train_test_val(
        all_data_path=os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'),
        all_embeddings_path=os.path.join(UNIPROT_DATA_DIR, 'all_embeddings.csv'),
        test_path=os.path.join(DETECT_DATA_DIR, 'test.csv'),
        train_path=os.path.join(DETECT_DATA_DIR, 'train.csv'),
        val_path=os.path.join(DETECT_DATA_DIR, 'val.csv'), **sizes)