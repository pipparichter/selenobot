
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import date
import re
import sklearn.cluster
from tqdm import tqdm
import random
import subprocess
import os
from os.path import join

from src.utils import *

DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/'
DETECT_DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/detect/'
FIGURE_DIR = '/home/prichter/Documents/selenobot/figures/'

# TODO: It might be worth writing a class to manage FASTA files. 


def truncate_sec(path=None, out_path=None, first_sec_only=True):
    '''Truncate the selenoproteins stored in the input file at each selenocysteine
    residue, sequentially.'''

    # Load the selenoproteins into a pandas DataFrame. 
    df = pd_from_fasta(path)

    df_trunc = {'id':[], 'seq':[]}
    for row in df.itertuples():
        # Find indices where a selenocysteine occurs. 
        idxs = np.where(np.array(list(row.seq)) == 'U')[0]

        # Only truncate at the first residue if that option is set. 
        if first_sec_only:
            idxs = idxs[0:1]

        # Sequentially truncate at each selenocysteine redidue. 
        seqs = [row.seq[:idx] for idx in idxs]
        # Add new truncated sequences to the dictionary.  
        df_trunc['id'] += [row.Index + f'[{i + 1}]' for i in range(len(seqs))]
        df_trunc['seq'] += seqs

    df_trunc = pd.DataFrame(df_trunc).set_index('id')
    pd_to_fasta(df_trunc, path=out_path)


def sample_kmeans_clusters(data, size, n_clusters=500, sec_ids=None):
    '''Sample n elements such that the elements are spread across a set
    of K-means clusters of. Returns the indices of data elements in the sample '''
    
    # kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init='auto')
    kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init='auto')
    kmeans.fit(data.values)

    # Sample from each cluster. 
    n_clusters = kmeans.cluster_centers_.shape[0]
    n = size // n_clusters # How much to sample from each cluster. 
    
    idxs = []
    for cluster in range(n_clusters):
        # Get indices for all data in a particular cluster, and randomly select n. 
        cluster_idxs = np.where(kmeans.labels_ == cluster)[0]
        idxs += list(np.random.choice(cluster_idxs, min(n, len(cluster_idxs))))
    # Maks sure no duplicate indices are collected. 

    # Make sure to exclude the IDs which are selenoproteins. 
    idxs = [idx for idx in idxs if data.index[idx] not in sec_ids]
    return np.unique(idxs)


def sample_sprot(path, out_path=None, size=None, sec_ids=None):
    '''Filters the SwissProt database by selecting representative proteins. The approach for doing this
    is based on K-means clustering the embedding data, and sampling from each cluster.

    args:
        - path (str): The path to the SwissProt embedding data.  
        - out_path (str): The path to write the filtered data to. 
        - size (int): The approximate number of items which should pass the filter.
        - sec_ids (np.array): The gene IDs of selenoproteins. 
    '''
    # Read in the FASTA file. 
    data = pd_from_fasta(path)

    # Read in the embeddings based on the filepath given as input. 
    directory, filename = os.path.split(path)
    embedding_path = os.path.join(directory, filename.split('.')[0] + '_embeddings.csv')
    embeddings = pd.read_csv(embedding_path, index_col='id')

    idxs = sample_kmeans_clusters(embeddings, size, sec_ids=sec_ids)
    
    # Filter the data, and write the filtered data as a FASTA file. 
    pd_to_fasta(data.iloc[idxs], path=out_path)


def add_embeddings(data, emb_path=None):
    '''Extract embeddings which match the gene IDs in the data DataFrame, and add them
    to the DataFrame.
    
    args:
        - data (pd.DataFrame): A pandas DataFrame with data from a FASTA file. 
            should have columns 'seq' and index 'id'.
        - emb_path (str): Path to the CSV file in which the embeddings are stored. 
    '''

    # Read in the embeddings from the specified path. 
    embeddings = pd.read_csv(emb_path)
    
    # Want to avoid loading in all the embeddings at once. 
    emb_ids = np.ravel(pd.read_csv(emb_path, usecols=['id']).values)
    
    idxs = np.isin(emb_ids, data.index)
    seqs = data['seq']

    data = pd.read_csv(emb_path, index_col='id')[idxs]
    # Make sure to add the sequence information back in. 
    data['seq'] = seqs

    return data


def add_labels(data, sec_ids=None):
    '''Add labels to the data stored at path. The labels are either 1 or 0, indicating
     whether or not the gene present in the dataset is a selenoprotein.'''

    labels = [1 if id_ in sec_ids else 0 for id_ in data.index]
    data['label'] = labels

    return data

# NOTE: Should I make the test set or the validation set smaller? I am just going to go with the validation set being smaller. 
def train_test_val_split(path, train_size=0.8):
    '''Splits the data stored in the input path into a train set, test set, and validation set.'''

    # Read in the data, and convert sizes to integers. 
    data = pd_from_fasta(path)

    train_size = int(train_size * len(data))
    # Because this is truncated, test_size should always be smaller than val_size. 
    test_size = (len(data) - train_size) // 2
    val_size = len(data) - train_size - test_size

    assert train_size + test_size + val_size == len(data), 'All subsets should sum to the total length of the dataset.'
    assert (train_size > val_size) and (val_size >= test_size), f'Expected size order is train_size > val_size >= test_size. Sizes are train_size={train_size}, val_size={val_size}, test_size={test_size}.'

    # Run CD-HIT on the data stored at the given path. 
    clusters = get_homology_clusters(path)

    # First want to split away the training data. 
    remainder, train_ids = split_by_homology(clusters, size=len(data) - train_size)

    # Filter the remaining clusters, and split into validation and test sets. 
    clusters = clusters[np.isin(clusters.index, remainder)]
    test_ids, val_ids = split_by_homology(clusters, size=test_size)

    assert len(np.unique(np.concatenate([train_ids, test_ids, val_ids]))) == len(np.concatenate([train_ids, test_ids, val_ids])), 'Some proteins are represented more than once in the partitioned data.'
    assert len(train_ids) + len(test_ids) + len(val_ids) == len(data), 'The lengths of the combined partitioned data do not add up to the original dataset.'

    # Use the obtained IDs to filter the dataset, and return the DataFrames. 
    train_data = data[np.isin(data.index, train_ids)]
    test_data = data[np.isin(data.index, test_ids)]
    val_data = data[np.isin(data.index, val_ids)]

    return train_data, test_data, val_data


def split_by_homology(clusters, size=None):
    '''Split the input data in two, ensuring that no sequences which are homologous 
    to one another are present in both groups.
    
    args:
        - clusters (pd.DataFrame): A pandas DataFrame mapping the gene ID to cluster number. 
        - size (int): The size of the first group, which must also be the smaller group. 
    '''
    assert (len(clusters) - size) >= size, f'The size argument must specify the smaller partition. Provided arguments are len(clusters)={len(lusters)} and size={size}.'

    ids = [[], []]
    # Sort the groups in descending order of length. 
    it = iter(sorted(clusters.groupby('cluster'), key=len, reverse=True))

    # Expected number of iterations is equal to the number of clusters. 
    n_clusters = len(np.unique(clusters['cluster']))
    pbar = tqdm(total=n_clusters, desc='split_by_homology')

    group = 0
    # NOTE: Why does checking the length of the larger group make it not work?
    # If we check for the larger size, it's basically the same as adding to each group in an alternating way. 
    while (x := next(it, None)) is not None:

        # Unpack the iterator. 'cluster' should be a pandas DataFrame, with the indices as gene IDs. 
        _, cluster = x
        pbar.update()

        if (len(ids[0]) < size) and (group == 0):
            ids[0] += list(cluster.index)
            group = 1
        else:
            ids[1] += list(cluster.index)
            group = 0
        
    pbar.close()

    assert len(ids[0]) + len(ids[1]) == len(clusters), f'The combined sizes of the partitions ({len(ids[0]) + len(ids[1])}) do not add up to the size of the original data ({len(clusters)}).'
    assert len(ids[0]) < len(ids[1]), 'The first set of IDs should be smaller than the second.'

    return tuple(ids)


def get_homology_clusters(path, c=0.8):
    '''Run CD-HIT on the FASTA file stored in the input path, generating the
    homology-based sequence similarity clusters.'''

    cmd = '/home/prichter/cd-hit-v4.8.1-2019-0228/cd-hit'
    directory, filename = os.path.split(path)
    filename = filename.split('.')[0] # Remove the extension from the filename. 
    
    o = join(directory, 'out') # Input to the -o option.
    f = join(directory, filename + '.clstr') # The new cluster filename.

    # This means I need to make sure to delete the files if I want to re-run. 
    # Only re-custer if the cluster file doesn't exist already. 
    if not os.path.isfile(f):

        # Run the CD-HIT command on data stored in the input path. 
        subprocess.run(f"{cmd} -i {path} -o {o} -c {c}", shell=True)

        # Remove the output file containing the representative sequences. 
        subprocess.run(f'rm {o}', shell=True)
        subprocess.run(f'mv {o}.clstr {f}', shell=True)

    df = pd_from_clstr(f)

    return df

# TODO: Flesh this out in a way that's more user-friendly and easier to debug.

def generate_detect_data(verbose=False):
    '''Creates a dataset for the detection classification task, which involves determining
    whether a protein, truncated or not, is a selenoprotein. The data consists of truncated and
    non-truncated selenoproteins, as well as a number of normal full-length proteins equal to all 
    selenoproteins.'''

    # truncate_sec(path=join(DATA_DIR, 'sec.fasta'), out_path=join(DETECT_DATA_DIR, 'sec_trunc.fasta'), first_sec_only=True)
    # Make sure to use IDs of truncated proteins. 
    sec_ids = fasta_ids(join(DETECT_DATA_DIR, 'sec_trunc.fasta'))

    # # Get the number of truncated selenoproteins. 
    # sec_size = fasta_size(join(DATA_DIR, 'sec.fasta')) 

    # # Get the number of truncated selenoproteins. 
    # sec_size = fasta_size(join(DATA_DIR, 'sec.fasta')) 

    # sample_sprot(join(DATA_DIR, 'sprot.fasta'), out_path=join(DETECT_DATA_DIR, 'sprot.fasta'), size=100000 - sec_size, sec_ids=sec_ids)
    
    # fasta_concatenate([join(DETECT_DATA_DIR, 'sec_trunc.fasta'), join(DETECT_DATA_DIR, 'sprot.fasta')], out_path=join(DETECT_DATA_DIR, 'all.fasta'))
    # sample_sprot(join(DATA_DIR, 'sprot.fasta'), out_path=join(DETECT_DATA_DIR, 'sprot.fasta'), size=100000 - sec_size, sec_ids=sec_ids)
    
    # fasta_concatenate([join(DETECT_DATA_DIR, 'sec_trunc.fasta'), join(DETECT_DATA_DIR, 'sprot.fasta')], out_path=join(DETECT_DATA_DIR, 'all.fasta'))

    # ctrain_data, test_data, and val_data should have sequence information. 
    train_data, test_data, val_data = train_test_val_split(join(DETECT_DATA_DIR, 'all.fasta'))

    # Add labels and embeddings to the data, and write to the file.
    for data, filename in zip([train_data, test_data, val_data], ['train.csv', 'test.csv', 'val.csv']):
        data = add_embeddings(data, emb_path=join(DATA_DIR, 'embeddings.csv'))
        data = add_labels(data, sec_ids=sec_ids)
        data.to_csv(join(DETECT_DATA_DIR, filename))


if __name__ == '__main__':

    # csv_concatenate([join(DATA_DIR, 'sec_trunc_embeddings.csv'), join(DATA_DIR, 'sprot_embeddings.csv')], out_path=join(DATA_DIR, 'embeddings.csv'))
    generate_detect_data()




# def h5_to_csv(path, batch_size=100):
#     '''Convert a data file in HD5 format to a CSV file.'''

#     file_ = h5py.File(path)
#     keys = list(file_.keys())

#     batches = [keys[i:min(i + batch_size, len(keys))] for i in range(0, len(keys), batch_size)]

#     header = True
#     for batch in tqdm(batches, desc='h5_to_csv'):
#         data = np.array([file_[k][()] for k in batch])
#         df = pd.DataFrame(data)
#         df['id'] = [get_id(k) for k in batch]

#         # df = df.set_index('id')
#         df.to_csv(path.split('.')[0] + '.csv', mode='a', header=header)

#         # After the first loop, don't write the headers to the file. 
#         header = False

#     file_.close()

