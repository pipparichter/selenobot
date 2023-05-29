'''
'''

import torch
from transformers import AutoTokenizer
from utils import fasta_to_df, get_id, clstr_to_df
import requests
import pandas as pd
import numpy as np
import tensorflow as tf


def generate_labels(data):
    '''
    Get labels which map each sequence in a dataset to a value indicating it is short (0) or truncated (1). 

    args:
        - data (pd.DataFrame): The sequence data, which must at least contain an 'id' column. 

    returns: torch.Tensor
    '''
    # Get the IDs for all selenoproteins from the sec_trunc.fasta file. 
    sec_ids = set(fasta_to_df('/home/prichter/Documents/protex/data/sec.fasta')['id'])

    labels = np.zeros(len(data), dtype=np.int8) # Specify integer type. 
    for i in range(len(data)): # Should not be prohibitively long.
        if data['id'].iloc[i] in sec_ids:
            labels[i] = 1
    # Should have shape (batch_size, )
    return torch.from_numpy(labels) # Convert to tensor upon return. 


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, data, load_labels=True, name='facebook/esm2_t6_8M_UR50D'):

        if load_labels: # Whether or not to load in labels for the data. 
            self.labels = generate_labels(data)
        else:
            self.labels = None
        
        tokenizer = AutoTokenizer.from_pretrained(name)
        # Tokenize the sequences so they align with the ESM model. This is a dictionary with keys 'input_ids' and 'attention_mask'
        # Each key maps to a two-dimensional tensor of size (batch_size, sequence length)
        self.encodings = tokenizer(list(data['seq']), padding=True, truncation=True, return_tensors='pt')

        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    
def truncate(seq):
    '''
    Truncate the input amino acid sequence at the first selenocysteine residue,
    which is denoted "U."
    '''
    idx = seq.find('U')
    return seq[:idx]


def truncate_selenoproteins(fasta_file):
    '''
    Truncate the selenoproteins stored in the inputted file at the first selenocysteine residue. 
    Overwrites the original file with the truncated sequence data. 

    kwargs:
        - fasta_file
    '''
    data = fasta_to_df(fasta_file)

    # Truncate the data at the first 'U' residue
    func = np.vectorize(truncate, otypes=[str])
    data['seq'] = data['seq'].apply(func)

    # Write the DataFrame in FASTA format. THIS OVERWRITES THE FILE.
    df_to_fasta(data, fasta_file)


def download_short_proteins(dist, fasta_file):
    '''
    Download short proteins from the UniProt database according to the length distribution of the truncated selenoproteins.
    '''
    delta = 2
    # There are about 20,000 selenoproteins listed, so try to get an equivalent number of short proteins. 
    n_lengths = 100 # The number of lengths to sample
    size = 200 # Number of sequences to grab from the database at a time. 

    with open(fasta_file, 'w', encoding='utf8') as f:
        # NOTE: Not really sure why the sample isn't one-dimensional. 
        sample = np.ravel(st.resample(size=n_lengths)) # Sample lengths from the distribution.
        for l in sample:
            # Link generated for the UniProt REST API.
            lower, upper = min(0, int(l) - delta), int(l) + delta

            url = f'https://rest.uniprot.org/uniprotkb/search?format=fasta&query=%28%28length%3A%5B{lower}%20TO%20{upper}%5D%29%29&size={size}'
            # Send the query to UniProt. 
            response = requests.get(url)

            if response.status_code == 200: # Indicates query was successful.
                f.write(response.text)
            else:
                raise RuntimeError(response.text)

    # Remove duplicates, as this seems to be an issue. 
    data = fasta_to_df(fasta_file)
    data = data.drop_duplicates(subset=['id'])
    df_to_fasta(data, fasta_file)


def sum_clusters(clusters):
    '''
    Calculate the total number of entries in a set of clusters. 

    args:
        - clusters (list): A list of two-tuples where the first element is the cluster ID, and the second element is the cluster size. 
    '''
    return sum([c[1] for c in clusters] + [0])


def train_test_split(data, test_size=0.25, train_size=0.75):
    '''
    Splits a sequence dataset into a training set and a test set. 

    args:   
        - data (pd.DataFrame): A DataFrame with, at minimum, columns 'seq', 'cluster', 'id'. 
    kwargs: 
        - test_size (float)
        - train_size (float)
    '''
    if train_size + test_size != 1.0:
        raise ValueError('Test and train sizes must sum to one.')

    # Challenge here is to split up the clusters such that the train and test proportions make sense, but 
    # all sequence belonging to the same cluster are in the same data group. Finding an exact (or the best) solution
    # is technically NP-hard, but I can use a greedy approximation. 
    cluster_data = data['cluster'].to_numpy()
    clusters = [(c, np.sum(cluster_data == c)) for c in np.unique(cluster_data)] # Each cluster and the number of entries it covers. 
    clusters = sorted(clusters, key=lambda x : -x[1]) # Sort the data according to number of entries in each cluster. Sort from large to small. 

    train_clusters, test_clusters = [], []
    # With the procedure below, there is a case where an infinite loop occurs -- if both training and test groups fill up, and
    # there is still a cluster which has not been allocated. 
    n = len(data) # Total number of things in the dataset. 
    i, i_prev = 0, 0
    while i < len(clusters):

        if sum_clusters(train_clusters) / n < train_size:
            train_clusters.append(clusters[i])
            i += 1 # Only move on to the next cluster if this was added. 

        if i == len(clusters): # Make sure we haven't hit the last cluster.
            break

        if sum_clusters(test_clusters) / n < test_size:
            test_clusters.append(clusters[i])
            i += 1
        
        # Prevent any infinite looping behavior. If an infinite loop begins, add all remaining clusters to the 
        # training set and break. 
        if i == i_prev:
            train_clusters += clusters[i:]
            break
        i_prev = i

    # Now, the clusters have been organized into the training and test sets. 
    # The cluster size information isn't relevant anymore, so remove this information, leaving only the cluster labels.
    train_clusters = [c[0] for c in train_clusters]
    test_clusters = [c[0] for c in test_clusters]

    test_data = data[data['cluster'].isin(test_clusters)]
    train_data = data[data['cluster'].isin(train_clusters)]
    
    return train_data, test_data # For now, just return the full DataFrames.


