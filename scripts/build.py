import sys, re, os, time, wget
from selenobot.utils import DATA_DIR
import pandas as pd
from typing import NoReturn, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import subprocess
import argparse
import logging

CDHIT = '/home/prichter/cd-hit-v4.8.1-2019-0228/cd-hit' # Path to the CD-HIT command.import logging




def download_embeddings(filename:str='embeddings.csv'):
    '''Download pre-generated PLM embeddings for sequences in uniprot.fasta. This file was creatae
     using the selenobot-embed.py script.'''
    if not (filename in os.listdir(DATA_DIR)):
        logging.info(f'Downloading PLM embeddings to {filename}.')
        url = 'https://storage.googleapis.com/selenobot-data/embeddings.csv' # Embeddings are stored in a Google Cloud bucket.
        wget.download(url, os.path.join(DATA_DIR, filename))


def download_data() -> NoReturn:
    download_swissprot()
    download_selenoproteins()
    download_embeddings()
    # Unzip the SwissProt sequence file into the data directory.
    subprocess.run(f"tar -xzf {os.path.join(DATA_DIR, 'sprot.fasta.tar.gz')} -C {DATA_DIR}", shell=True, check=True)
    subprocess.run(f"gzip -d {os.path.join(DATA_DIR, 'uniprot_sprot.fasta.gz')}", shell=True, check=True)
    subprocess.run(f"mv {os.path.join(DATA_DIR, 'uniprot_sprot.fasta')} {os.path.join(DATA_DIR, 'sprot.fasta')}", shell=True, check=True)


    # # Modify the format of the FASTA file to standardize reading and writing from FASTA files.
    # for filename in ['sprot.fasta', 'sec.fasta']:
    #     path = os.path.join(DATA_DIR, filename)
    #     fasta = ''
    #     with open(path, 'r') as f:
    #         lines = f.readlines()
    #     for line in lines:
    #         if '>' in line: # This symbol marks the beginning of a header file.
    #             id_ = re.search('\|([\w\d_]+)\|', line).group(1)
    #             fasta += f'>id={id_}\n'
    #         else:
    #             fasta += line
    #     with open(path, 'w') as f:
    #         f.write(fasta) # This will overwrite the original file. 
    
    # # Remove known selenoproteins from the SwissProt file, so there is no data leakage. 
    # sprot_df = dataframe_from_fasta(os.path.join(DATA_DIR, 'sprot.fasta'))
    # sec_idxs = sprot_df.seq.str.contains('U')
    # logging.info(f'\tNumber of selenoproteins in downloaded SwissProt data: {np.sum(sec_idxs.values)}')
    # logging.info(f'\t{len(sprot_df) - len(sprot_df[~sec_idxs])} selenoproteins removed from sprot.fasta.')
    # dataframe_to_fasta(sprot_df[~sec_idxs], os.path.join(DATA_DIR, 'sprot.fasta')) # Overwrite the original FASTA file. 


def truncate_selenoproteins(filename:str='sec.fasta') -> NoReturn:
    '''Truncate the selenoproteins stored in the input file. This function assumes that all 
    sequences contained in the file contain selenocysteine, labeled as U.'''
    logging.info(f'Truncating selenoproteins in {filename}.')

    df = dataframe_from_fasta(os.path.join(DATA_DIR, filename)) # Load the selenoproteins into a pandas DataFrame.
    df_trunc = {'id':[], 'seq':[], 'n_trunc':[], 'n_sec':[]}
    for row in df.itertuples():
        df_trunc['id'].append(row.id + '[1]') # Modify the row ID to contain a [1] label. 
        first_sec_idx = row.seq.index('U') # This will raise an exception if no U residue is found.
        df_trunc['seq'].append(row.seq[:first_sec_idx]) # Get the portion of the sequence prior to the U residue.
        df_trunc['n_trunc'].append(len(row.seq) - first_sec_idx) # Store the number of amino acid residues discarded.
        df_trunc['n_sec'].append(row.seq.count('U')) # Store the number of selenoproteins in the original sequence.
    df_trunc = pd.DataFrame(df_trunc)

    logging.info(f'\tAverage number of selenocysteines per protein: {df_trunc.n_sec.mean()}')
    logging.info(f'\tMean length of truncation: {df_trunc.n_trunc.mean()}')
    logging.info(f'\tLargest truncation: {df_trunc.n_trunc.max()}')
    logging.info(f'\tSmallest truncation: {df_trunc.n_trunc.min()}')

    # Make sure there are no leftover selenocysteines. 
    assert np.all(~df_trunc.seq.str.contains('U')), 'truncate_selenoproteins: There are selenocysteine residues present in the set of truncated selenoproteins.'

    # Write the truncated proteins to a new file. 
    dataframe_to_fasta(df_trunc, path=filename.split('.')[0] + '_truncated.fasta')


def run_cdhit(filename:str='uniprot.fasta', n:int=5, c:float=0.8, l:int=5) -> NoReturn:
    '''Run the CD-HIT clustering tool on the data stored in the uniprot.fasta file.'''
    logging.info(f'Running CD-HIT on sequences in {filename}.')
    t1 = time.perf_counter()

    i = os.path.join(DATA_DIR, filename)
    o = os.path.join(DATA_DIR, filename.split('.')[0])
    # Run the CD-HIT command with the specified cluster parameters. 
    subprocess.run(f'{CDHIT} -i {i} -o {o} -n {n} -c {c} -l {l}', shell=True, check=True, stdout=subprocess.DEVNULL)

    t2 = time.perf_counter()
    logging.info(f'CD-HIT run completed in {np.round(t2 - t1, 2)} seconds.')
    # Print some information about the clustering process. 
    clstr_df = dataframe_from_clstr(os.path.join(DATA_DIR, 'uniprot.clstr'))
    fasta_df = dataframe_from_fasta(os.path.join(DATA_DIR, filename))
    logging.info(f'\tNumber of clusters: {len(set(clstr_df.cluster.values))}')

    cluster_sizes = clstr_df.groupby('cluster').apply(len, include_groups=False)
    logging.info(f'\tMean cluster size: {cluster_sizes.mean()}')
    logging.info(f'\tMinimum cluster size: {cluster_sizes.min()}')
    logging.info(f'\tMaximum cluster size: {cluster_sizes.max()}')
    # logging.info(f'\tNumber of sequences not assigned clusters: {len(fasta_df) - len(clstr_df)}')


def split_data(fasta_filename:str='uniprot.fasta', clstr_filename:str='uniprot.clstr') -> NoReturn:
    '''Divide the uniprot data into training, testing, and validation datasets. The split is cluster-aware, and ensures that
    no CD-HIT-generated cluster is split between datasets. The data is first divided into a training set (80 percent) and test
    set (20 percent), and the training data is further divided into a training (80 percent) and validation (20 percent) set.'''
    
    logging.info('Dividing data into training, testing, and validation sets.')
    clstr_df = dataframe_from_clstr(os.path.join(DATA_DIR, clstr_filename)).set_index('id') # Has columns id and cluster.
    fasta_df = dataframe_from_fasta(os.path.join(DATA_DIR, fasta_filename)).set_index('id')
    
    n1 = len(fasta_df) # Get original number of sequences in the dataset. 
    fasta_df, clstr_df = fasta_df.align(clstr_df, join='inner', axis=0) # Used an intersection of the IDs present in both DataFrames. 
    n2 = len(fasta_df) # Get size of data after alignment. 
    logging.info(f'\tNumber of sequences dropped from {fasta_filename}: {n1 - n2}')
    groups = clstr_df['cluster'].values # Extract cluster labels. 

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8)
    idxs, test_idxs = list(gss.split(fasta_df.values, groups=groups))[0]
    test_ids = fasta_df.index[test_idxs] # Get the IDs for entries in the test set. 

    fasta_df.iloc[test_idxs].to_csv(os.path.join(DATA_DIR, 'test.csv'))
    test_size = len(test_idxs)
    
    fasta_df, groups = fasta_df.iloc[idxs], groups[idxs] # Now working only with the remaining sequence data, not in the test set. 
    train_idxs, val_idxs = list(gss.split(fasta_df.values, groups=groups))[0]
    train_ids, val_ids = fasta_df.index[train_idxs], fasta_df.index[val_idxs] # Get the IDs for entries in the training and validation sets. 

    fasta_df.iloc[train_idxs].to_csv(os.path.join(DATA_DIR, 'train.csv'))
    fasta_df.iloc[val_idxs].to_csv(os.path.join(DATA_DIR, 'val.csv'))
    train_size, val_size = len(train_idxs), len(val_idxs)

    logging.info(f'\tSize of training dataset: {train_size}')
    logging.info(f'\tSize of testing dataset: {test_size}')
    logging.info(f'\tSize of validation dataset: {val_size}')

    # Make sure everything worked correctly...
    train_groups = set(clstr_df.loc[train_ids].cluster.values) # Get the groups represented in the training set. 
    val_groups = set(clstr_df.loc[val_ids].cluster.values) # Get the groups represented in the validation set. 
    test_groups = set(clstr_df.loc[test_ids].cluster.values) # Get the groups represented in the testing set. 
    assert len(train_groups.intersection(val_groups).intersection(test_groups)) == 0, 'split_data: Detected leakage between datasets.'
    assert (train_size + test_size + val_size) == n2, 'split_data: Some data was lost during the train-test-val split.'


def add_embeddings(filename:str, chunk_size:int=1000) -> NoReturn:
    '''Add embedding information to a dataset, and overwrite the original dataset with the
    modified dataset (with PLM embeddings added).
    
    :param path: The path to the dataset.
    :chunk_size: The size of the chunks to split the dataset into for processing.
    '''
    logging.info(f'Adding PLM embeddings to {filename}.')

    embeddings_path = os.path.join(DATA_DIR, 'embeddings.csv') # Path to the PLM embeddings, 
    dataset_path = os.path.join(DATA_DIR, filename) # Path to the dataset to which to add embeddings. 
    tmp_path = os.path.join(DATA_DIR, 'tmp.csv') # The path to the temporary file to which the modified dataset will be written in chunks.

    embedding_ids = pd.read_csv(embeddings_path, usecols=['id'])['id'].values.ravel() # Read the IDs in the embedding file to avoid loading the entire thing into memory.
    reader = pd.read_csv(dataset_path, index_col=['id'], chunksize=chunk_size) # Use read_csv to load the dataset one chunk at a time. 

    is_first_chunk = True
    for chunk in reader:
        # Get the indices of the embedding rows corresponding to the data chunk. Make sure to shift the index up by one to account for the header. 
        idxs = np.where(np.isin(embedding_ids, chunk.index, assume_unique=True))[0] + 1 
        idxs = [0] + list(idxs) # Add the header index so the column names are included. 
        # Read in the embedding rows, skipping rows which do not match a gene ID in the chunk. 
        chunk = chunk.merge(pd.read_csv(embeddings_path, skiprows=lambda i : i not in idxs), on='id', how='inner')
        # Check to make sure the merge worked as expected. Subtract 1 from len(idxs) to account for the header row.
        assert len(chunk) == (len(idxs) - 1), f'add_embeddings: Data was lost while merging embedding data with {filename}.'
        
        chunk.to_csv(tmp_path, header=is_first_chunk, mode='w' if is_first_chunk else 'a') # Only write the header for the first file. 
        is_first_chunk = False

    # Replace the old dataset with the temporary file. 
    subprocess.run(f'rm {dataset_path}', shell=True, check=True)
    subprocess.run(f'mv {tmp_path} {dataset_path}', shell=True, check=True)

if __name__ == '__main__':

    t1 = time.perf_counter()

    if not args.skip_download:
        download_data()

    truncate_selenoproteins()

    # Combine the sec_truncated.fasta and sprot.fasta files into a single uniprot.fasta file. 
    uniprot_path, sec_truncated_path, sprot_path = os.path.join(DATA_DIR, 'uniprot.fasta'), os.path.join(DATA_DIR, 'sec_truncated.fasta'), os.path.join(DATA_DIR, 'sprot.fasta')
    subprocess.run(f'cat {sec_truncated_path} {sprot_path} > {uniprot_path}', shell=True, check=True)
    
    if not args.skip_cdhit:
        run_cdhit()

    split_data()

    for filename in ['train.csv', 'test.csv', 'val.csv']:
        df = pd.read_csv(os.path.join(DATA_DIR, filename))
        # Add a label column, marking those sequences with a [1] in the gene ID with a 1.
        # Make sure to turn off regex for the str.contains function, or it will try to match [1] as a pattern.
        df['label'] = df['id'].str.contains('[1]', regex=False).astype(int)
        df.set_index('id').to_csv(os.path.join(DATA_DIR, filename))
    
    for filename in ['train.csv', 'test.csv', 'val.csv']:
        add_embeddings(filename)

    t2 = time.perf_counter()
    logging.info(f'Selenobot set-up complete in {np.round(t2 - t1, 2)} seconds.')