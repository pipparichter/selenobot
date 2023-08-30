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
    # df.index = df.index.astype(str)
    df.index = df.index.map(str)

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


def sample_kmeans_clusters(data, size, n_clusters=500):
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

    idxs = sample_kmeans_clusters(embeddings, size)
    # Make sure to exclude the IDs which are selenoproteins. 
    idxs = [idx for idx in idxs if data.index[idx] not in sec_ids]

    # Filter the data, and write the filtered data as a FASTA file. 
    pd_to_fasta(data.iloc[idxs], path=out_path)


def train_test_split(path, train_size=None):
    '''Split the sequence data into a training and test set.'''

    # Run CD-HIT on the data stored at the given path. 
    clusters = get_homology_clusters(path)
   
    # First, join the cluster information and data on the gene IDs (indies.). 
    data = pd_from_fasta(path)
    data = data.join(clusters)

    test_data, train_data = pd.DataFrame(columns=data.columns) , pd.DataFrame(columns=data.columns)

    # Sort the groups in descending order of length. 
    it = iter(sorted(data.groupby('cluster'), key=len, reverse=True))
    # Expected number of iterations is equal to the number of clusters. 
    pbar = tqdm(total=len(np.unique(data['cluster'])), desc='train_test_split')

    while (x := next(it, None)) is not None:
        # Unpack the iterator. 
        _, cluster = x
        pbar.update()

        if len(train_data) < train_size:
            train_data = pd.concat([cluster, train_data])
        else:
            test_data = pd.concat([cluster, test_data])
        
        if (x := next(it, None)) is not None:
            # Unpack the iterator. 
            _, cluster = x
            pbar.update()

            test_data = pd.concat([cluster, test_data])

    pbar.close()

    train_data.index.name = 'id'
    test_data.index.name = 'id'

    return train_data, test_data


def get_homology_clusters(path, c=0.8):
    '''Run CD-HIT on the FASTA file stored in the input path, generating the
    homology-based sequence similarity clusters.'''

    cmd = '/home/prichter/cd-hit-v4.8.1-2019-0228/cd-hit'
    directory, filename = os.path.split(path)
    filename = filename.split('.')[0] # Remove the extension from the filename. 

    o = join(directory, 'out') # Input to the -o option.
    f = join(directory, filename + '.clstr') # The new cluster filename.

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

    # define the sub-directory of the data directory for this particular dataset. 
    sub_data_dir = join(data_dir, 'detect/')

    # Get the number of truncated and non-truncated selenoproteins. 
    size = fasta_size(data_dir + 'sec_trunc.fasta') + fasta_size(data_dir + 'sec.fasta')
    filter_sprot(join(data_dir, 'sprot.fasta'), out_path=join(sub_data_dir, 'sprot.fasta'), size=size, sec_ids=fasta_ids(join(data_dir, 'sec.fasta')))
    
    path = join(sub_data_dir, 'all.fasta') # Path to write all the data to. 
    paths = [join(data_dir, 'sec_trunc.fasta'), join(sub_data_dir, 'sprot.fasta'), join(data_dir, 'sec.fasta')]
    fasta_concatenate(paths, out_path=path)

    # Want do do something like a 70-30 train-test split. 
    train_size = int(0.7 * fasta_size(path))
    train_data, test_data = train_test_split(path, train_size=train_size)

    # Write the data to files. 
    train_data.to_csv(join(sub_data_dir, 'train.csv'))
    test_data.to_csv(join(sub_data_dir, 'test.csv'))

    plot_train_test_split(train_data, test_data, path=join(figure_dir, 'train_test_split_detect.png'))
 

def generate_extend_data(data_dir, figure_dir):
    '''Creates a dataset for the detection classification task, which involves determining
    whether a selenoprotein is truncated or not.''' 

    # define the sub-directory of the data directory for this particular dataset. 
    sub_data_dir = join(data_dir, 'extend/')

    path = join(sub_data_dir, 'all.fasta') # Path to write all the data to. 
    paths = [join(data_dir, 'sec_trunc.fasta'), join(data_dir, 'sec.fasta')]
    fasta_concatenate(paths, out_path=path)

    # Want do do something like a 70-30 train-test split. 
    train_size = int(0.7 * fasta_size(path))
    train_data, test_data = train_test_split(path, train_size=train_size)

    # Write the data to files. 
    train_data.to_csv(join(sub_data_dir, 'train.csv'))
    test_data.to_csv(join(sub_data_dir, 'test.csv'))

    plot_train_test_split(train_data, test_data, path=join(figure_dir, 'train_test_split_extend.png'))
    
 
def generate_combo_data(data_dir, figure_dir):
    '''Creates a dataset for the combination classification task (both flagging a candidate selenoprotein and
    predicting whether or not it is truncated). This dataset consists of full-length proteins, full-length
    selenoproteins, and truncated selenoproteins.'''

    # define the sub-directory of the data directory for this particular dataset. 
    sub_data_dir = join(data_dir, 'combo/')

    # Grab a number of full-length proteins roughly equal to the number of truncated selenoproteins. 
    size = fasta_size(data_dir + 'sec_trunc.fasta') 
    filter_sprot(join(data_dir, 'sprot.fasta'), out_path=join(sub_data_dir, 'sprot.fasta'), size=size, sec_ids=fasta_ids(join(data_dir, 'sec.fasta')))
    
    path = join(sub_data_dir, 'all.fasta') # Path to write all the data to. 
    paths = [join(data_dir, 'sec_trunc.fasta'), join(sub_data_dir, 'sprot.fasta'), join(data_dir, 'sec.fasta')]
    fasta_concatenate(paths, out_path=path)

    # Want do do something like a 70-30 train-test split. 
    train_size = int(0.7 * fasta_size(path))
    train_data, test_data = train_test_split(path, train_size=train_size)

    # Write the data to files. 
    train_data.to_csv(join(sub_data_dir, 'train.csv'))
    test_data.to_csv(join(sub_data_dir, 'test.csv'))

    plot_train_test_split(train_data, test_data, path=join(figure_dir, 'train_test_split_combo.png'))
    

if __name__ == '__main__':

    # truncate_sec(path=join(DATA_DIR, 'sec.fasta'), out_path=join(DATA_DIR, 'sec_trunc.fasta'))
    
    # generate_detect_data(DATA_DIR, FIGURE_DIR)
    # generate_extend_data(DATA_DIR, FIGURE_DIR)
    generate_combo_data(DATA_DIR, FIGURE_DIR)



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

