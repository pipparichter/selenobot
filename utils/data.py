'''
Utility functions for gathering and preprocessing data.
'''

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import date
import re
import io
import h5py
import sklearn.cluster
from tqdm import tqdm
import random
import subprocess
import os
from os.path import join

import sys
sys.path.append('/home/prichter/Documents/selenobot/utils')

from plot import *

DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/' 
FIGURE_DIR = '/home/prichter/Documents/selenobot/figures/'

# TODO: It might be worth writing a class to manage FASTA files. 

def write(text, path):
    '''Writes a string of text to a file.'''
    if path is not None:
        with open(path, 'w') as f:
            f.write(text)

def read(path):
    '''Reads the information contained in a text file into a string.'''
    with open(path, 'r') as f:
        text = f.read()
    return text


def clear(path):
    '''Clear the contents of the file found at the path.'''
    open(path, 'w').close()


def get_id(head):
    '''Extract the unique identifier from a FASTA metadata string (the 
    information on the line preceding the actual sequence). This ID should 
    be flanked by '|'.
    '''
    start_idx = head.find('|') + 1
    # Cut off any extra stuff preceding the ID, and locate the remaining |.
    head = head[start_idx:]
    end_idx = head.find('|')
    return head[:end_idx]


def fasta_ids(path):
    '''Extract all gene IDs stored in a FASTA file.'''
    # Read in the FASTA file as a string. 
    fasta = read(path)
    # Extract all the IDs from the headers, and return the result. 
    ids = [get_id(head) for head in re.findall(r'^>.*', fasta, re.MULTILINE)]
    return np.array(ids)

def csv_ids(path):
    '''Extract all gene IDs stored in a CSV file.'''
    df = pd.read_csv(path, usecols=['id']) # Only read in the ID values. 
    return np.ravel(df.values)

def csv_labels(path):
    '''Extract all gene IDs stored in a CSV file.'''
    df = pd.read_csv(path, usecols=['label']) # Only read in the ID values. 
    # Seems kind of bonkers that I need to ravel this. 
    return np.ravel(df.values.astype(np.int32))


def fasta_seqs(path):
    '''Extract all amino acid sequences stored in a FASTA file.'''
    # Read in the FASTA file as a string. 
    fasta = read(path)
    seqs = re.split(r'^>.*', fasta, flags=re.MULTILINE)[1:]
    # Strip all of the newline characters from the amino acid sequences. 
    seqs = [s.replace('\n', '') for s in seqs]
    # return np.array(seqs)
    return seqs
    

def fasta_size(path):
    '''Get the number of entries in a FASTA file.'''
    return len(fasta_ids(path))


def fasta_concatenate(paths, out_path=None):
    '''Combine the FASTA files specified by the paths. Creates a new file
    containing the combined data.'''

    dfs = [pd_from_fasta(p) for p in paths]
    df = pd.concat(dfs)
    pd_to_fasta(df, path=out_path)


def csv_concatenate(paths, out_path=None):
    '''Combine the CSV files specified by the paths. Creates a new file
    containing the combined data.'''

    dfs = [pd.read_csv(p, index_col='id') for p in paths]
    df = pd.concat(dfs)
    df.to_csv(out_path)


def pd_from_clstr(path):
    '''Convert a .clstr file string to a pandas DataFrame. The resulting 
    DataFrame maps cluster label to gene ID.'''

    # Read in the cluster file as a string. 
    clstr = read(path)
    df = {'id':[], 'cluster':[]}
    # The start of each new cluster is marked with a line like ">Cluster [num]"
    clusters = re.split(r'^>.*', clstr, flags=re.MULTILINE)
    # Split on the newline. 
    for i, cluster in enumerate(clusters):
        ids = [get_id(x) for x in cluster.split('\n') if x != '']
        df['id'] += ids
        df['cluster'] += [i] * len(ids)

    return pd.DataFrame(df).set_index('id')


def pd_from_fasta(path):
    '''Load a FASTA file in as a pandas DataFrame.'''

    ids = fasta_ids(path)
    seqs = fasta_seqs(path)

    df = pd.DataFrame({'seq':seqs, 'id':ids})
    # df = df.astype({'id':str, 'seq':str})
    df = df.set_index('id')
    
    return df


def pd_to_fasta(df, path=None, textwidth=80):
    '''Convert a pandas DataFrame containing FASTA data to a FASTA file format.'''

    fasta = ''
    for row in tqdm(df.itertuples(), desc='df_to_fasta', total=len(df)):
        fasta += '>|' + str(row.Index) + '|\n'

        # Split the sequence up into shorter, sixty-character strings.
        n = len(row.seq)
        seq = [row.seq[i:min(n, i + textwidth)] for i in range(0, n, textwidth)]

        seq = '\n'.join(seq) + '\n'
        fasta += seq
    
    # Write the FASTA string to the path-specified file. 
    write(fasta, path=path)


def filter_sec_trunc(path, trunc=1, out_path=None):
    '''Scans the file containing the sequentially-truncated selenoproteins, and grabs those
    proteins which have been truncated at the [trunc]th Sec residue.
    
    args:
        - path (str): Path to the file containing the truncated selenoproteins. 
        - trunc (int): Number of the Sec residue at which the collected proteins
            are truncated. 
    '''
    tag = f'[{trunc}]'
    df = pd_from_fasta(path)
    idxs = np.array([tag in id_ for id_ in df.index])

    # Filter numpy array by Boolean index. 
    df = df[idxs]
    pd_to_fasta(df, path=out_path)



def truncate_sec(path=None, out_path=None):
    '''Truncate the selenoproteins stored in the input file at each selenocysteine
    residue, sequentially.'''

    # Load the selenoproteins into a pandas DataFrame. 
    df = pd_from_fasta(path)

    df_trunc = {'id':[], 'seq':[]}
    for row in df.itertuples():
        # Find indices where a selenocysteine occurs. 
        idxs = np.where(np.array(list(row.seq)) == 'U')[0]
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

    # Also return the fitted kmeans model. 
    plot_sample_kmeans_clusters(data, kmeans, n_clusters=n_clusters, sec_ids=sec_ids, path=join(FIGURE_DIR, 'sample_kmeans_clusters.png'))
    
    # Make sure to exclude the IDs which are selenoproteins. 
    idxs = [idx for idx in idxs if data.index[idx] not in sec_ids]
    return np.unique(idxs)


def filter_sprot(path, out_path=None, size=None, sec_ids=None):
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


def filter_embeddings(path, out_path=None, ids=None):
    '''Extract embeddings which match the specified gene IDs and write them to a new
    file at out_path.'''

    # Want to avoid loading in all the embeddings at once. 
    emb_ids = np.ravel(pd.read_csv(path, usecols=['id']).values)
    
    idxs = np.isin(emb_ids, ids)
    df = pd.read_csv(path, index_col='id')[idxs]
    df.to_csv(out_path)


def label(path, sec_ids=None):
    '''Add labels to the data stored at path. The labels are either 1 or
    0, indicating whether or not the gene present in the dataset is 
    a selenoprotein or not.'''

    df = pd.read_csv(path, index_col='id')
    labels = [1 if id_ in sec_ids else 0 for id_ in df.index]
    
    df['label'] = labels
    df.to_csv(path)


def train_test_val_split(path, train_size=0.8, test_size=0.1, val_size=0.1):
    '''Splits the data stored in the input path into a train set, test set, and validation set.'''

    # Read in the data, and convert sizes to integers. 
    data = pd_from_fasta(path)

    train_size = int(train_size * len(data))
    # Because this is truncated, test_size should always be smaller than val_size. 
    test_size = int(test_size * len(data))
    val_size = len(data) - train_size - test_size

    assert train_size + test_size + val_size == len(data)

    # Run CD-HIT on the data stored at the given path. 
    clusters = get_homology_clusters(path)

    # First want to split away the training data. 
    the_rest, train_ids = split_by_homology(clusters, size=len(data) - train_size)

    # Filter the remaining clusters, and split into validation and test sets. 
    clusters = clusters[np.isin(clusters.index, the_rest)]
    test_ids, val_ids = split_by_homology(clusters, size=test_size)

    print(len(train_ids), train_size)
    print(len(test_ids), test_size)
    print(len(val_ids), val_size)

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
    assert (len(clusters) - size) > size

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


def generate_detect_data(data_dir, figure_dir):
    '''Creates a dataset for the detection classification task, which involves determining
    whether a protein, truncated or not, is a selenoprotein. The data consists of truncated and
    non-truncated selenoproteins, as well as a number of normal full-length proteins equal to all 
    selenoproteins.'''

    # Define the sub-directory of the data directory for this particular dataset. 
    sub_data_dir = join(data_dir, 'detect/')

    path = join(sub_data_dir, 'all.fasta') # Path to write all the data to. 
    paths = [join(sub_data_dir, 'sec_trunc.fasta'), join(sub_data_dir, 'sprot.fasta')]

    # # Get the number of truncated and non-truncated selenoproteins. 
    # size = fasta_size(data_dir + 'sec_trunc.fasta') # + fasta_size(data_dir + 'sec.fasta')
    sec_ids = fasta_ids(join(data_dir, 'sec.fasta'))
    # filter_sprot(join(data_dir, 'sprot.fasta'), out_path=join(sub_data_dir, 'sprot.fasta'), size=100000-size, sec_ids=sec_ids)
    
    # filter_sec_trunc(join(data_dir, 'sec_trunc.fasta'), trunc=1, out_path=join(sub_data_dir, 'sec_trunc.fasta'))
    # fasta_concatenate(paths, out_path=path)


    # train_data, test_data, val_data = train_test_val_split(path)

    # filter_embeddings(join(data_dir, 'embeddings.csv'), out_path=join(sub_data_dir, 'train.csv'), ids=train_data.index)
    # filter_embeddings(join(data_dir, 'embeddings.csv'), out_path=join(sub_data_dir, 'test.csv'), ids=test_data.index)
    # filter_embeddings(join(data_dir, 'embeddings.csv'), out_path=join(sub_data_dir, 'val.csv'), ids=val_data.index)

    # # Add labels to the embedding data.
    # label(join(sub_data_dir, 'train.csv'), sec_ids=sec_ids)
    # label(join(sub_data_dir, 'test.csv'), sec_ids=sec_ids)
    # label(join(sub_data_dir, 'val.csv'), sec_ids=sec_ids)


    # plot_train_test_val_split(train_data, test_data, val_data, path=join(figure_dir, 'train_test_val_split_detect.png'))

    print(len(sec_ids) / fasta_size(path))

 


if __name__ == '__main__':

    # csv_concatenate([join(DATA_DIR, 'sec_trunc_embeddings.csv'), join(DATA_DIR, 'sprot_embeddings.csv')], out_path=join(DATA_DIR, 'embeddings.csv'))
    # truncate_sec(path=join(DATA_DIR, 'sec.fasta'), out_path=join(DATA_DIR, 'sec_trunc.fasta'))
    
    generate_detect_data(DATA_DIR, FIGURE_DIR)
    # generate_extend_data(DATA_DIR, FIGURE_DIR)




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

