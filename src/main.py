'''
Main body of code for the ProtEx project. Possibly divide across multiple modules in the future. 
'''

# Silence all the deprecation warnings being triggered by umap-learn.
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore') 

import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import umap
from tqdm import tqdm
import scipy.stats as stats
import requests


DATA_DIR = '/home/prichter/Documents/protex/data/'
EMBEDDINGS_DIR = '/home/prichter/Documents/protex/data/embeddings/'
CLUSTER_DIR = '/home/prichter/Documents/protex/data/clusters/'

def clstr_to_df(read_from='all.clstr'):
    '''
    Reads a clustr file (the output file from CD-hit) into a pandas DataFrame.

    kwargs:
        : read_from (str)
    '''
    cluster_count = 0
    df = {'cluster':[], 'cluster_idx':[], 'is_rep':[], 'percent_similarity':[], 'id':[]}

    with open(CLUSTER_DIR + read_from, 'r', encoding='utf8') as f:
        lines = f.readlines()
        i = 1 # Skip the first line, which is just a cluster header. 

        while i < len(lines):
            line = lines[i]
            if 'Cluster' not in line:
                line = line.split() # Split along whitespace.
                
                df['cluster'].append(cluster_count)
                df['cluster_idx'].append(int(line[0]))

                if line[-1] == '*': # If the last character is an asterisk, then it's a representative sequence.
                    df['is_rep'].append(True)
                    df['percent_similarity'].append(100) # 100 percent similarity to the representative sequence.
                else: # If the sequence is not representative, it ends in '... at [x]%'
                    df['is_rep'].append(False)
                    df['percent_similarity'].append(float(line[-1][:-1]))
                
                # In the clustr file, the start point of the unique ID is given by >.
                df['id'].append(get_id(line[2]))
            else: # Move on to the next cluster. 
                cluster_count += 1

            i += 1 # Increment the line number, regardless of whether or not the cluster count is incremented. 

    # Convert to a DataFrame and return. Make sure datatypes are correct.  
    return pd.DataFrame(df).astype({'cluster':'int64', 'id':'str'})


def get_id(metadata):
    '''
    Extract the unique identifier from a FASTA metadata string (the information on the line preceding
    the actual sequence). This ID should be flanked by '|'.

    args:
        : metadata (str)
    kwargs:
        : start (str)
        : end (str)
    '''
    start_idx = metadata.find('|') + 1
    # Cut off any extra stuff preceding the ID, and locate the remaining |.
    metadata = metadata[start_idx:]
    end_idx = metadata.find('|')

    return str(metadata[:end_idx])


def fasta_to_df(read_from='sec_full.fasta'):
    '''
    Read data from a FASTA file and convert it to a pandas DataFrame.
    '''
    df = {'seq':[], 'metadata':[], 'id':[]}

    # Open the FASTA file. Metadata is contained on a single line, with the sequence given on the subsequent lines.
    # The sequence is broken up across multiple lines, with each line containing 60 characters.
    with open(DATA_DIR + read_from, 'r', encoding='utf8') as f:
        lines = f.readlines()

        i = 0 
        while i < len(lines):
            # I think this line should include the newline character.
            df['metadata'].append(lines[i])
            # Also extract the unique ID from the metadata line, and add to the DataFrame. 
            df['id'].append(get_id(lines[i]))
            i += 1

            seq = ''
            # Read in the amino acid sequence. The next metadata line wih have a '>' in the front.
            while (i < len(lines)) and (lines[i][0] != '>'):
                # Remove whitespace from sequence.
                seq += lines[i].strip()
                i += 1        
            df['seq'].append(seq)

    # Convert to a DataFrame and return. 
    return pd.DataFrame(df).astype({'id':'str'})


def embedding_to_txt(embedding, write_to='sec_trunc.txt', mode='a'):
    '''
    Writes an embedding, which is given in the form of a tensor, to a text file in the 
    embeddings directory. Note that this function appends to the file by default -- it does not overwrite,
    as embeddings need to be generated in chunks. 

    args:
        : embedding (np.ndarray)
    kwargs:
        : write_to (str): The file to write to. 
    '''
    
    with open(EMBEDDINGS_DIR + write_to, mode, encoding='utf8') as f: 
        np.savetxt(f, embedding.detach().numpy(), delimiter=' ', newline='\n')


def df_to_fasta(df, write_to='sec_full.fasta'):
    '''
    Write a DataFrame containing FASTA data to a FASTA file format.
    '''
    with open(DATA_DIR + write_to, 'w', encoding='utf8') as f:
        # Iterate over the DataFrames as a set of named tuples.
        for row in df.itertuples():
            f.write(row.metadata)
            
            # Split the sequence up into shorter, sixty-character strings.
            n = len(row.seq)
            seq = [row.seq[i:min(n, i + 60)] for i in range(0, n, 60)]
            seq = '\n'.join(seq) + '\n'
            f.write(seq)


def truncate(seq):
    '''
    Truncate the input amino acid sequence at the first selenocysteine residue,
    which is denoted "U."
    '''
    idx = seq.find('U')
    return seq[:idx]


def truncate_selenoproteins(read_from='../data/sec_full.fasta', write_to='../data/sec_trunc.fasta'):
    '''
    Truncate the selenoproteins stored in the read_from file at the first selenocysteine residue. 

    kwargs:
        : read_from (str)
        : write_to (str)
    '''
    data = fasta_to_df(read_from=read_from)

    # Truncate the data at the first 'U' residue
    func = np.vectorize(truncate, otypes=[str])
    data['seq'] = data['seq'].apply(func)

    # Write the DataFrame in FASTA format. 
    df_to_fasta(data, write_to=write_to)


def download_short_proteins(read_from='sec_trunc.fasta', write_to='short.fasta'):
    '''
    Download short proteins from the UniProt database according to the length distribution of the truncated selenoproteins.
    '''
    # First, read in the selenoprotein data and get the lengths.
    data = fasta_to_df(read_from=read_from)
    lengths = data['seq'].apply(len).to_numpy()
    lengths_kde = stats.gaussian_kde(lengths)

    # There are about 20,000 selenoproteins listed, so try to get an equivalent number of short proteins. 
    n_lengths = 100 # The number of lengths to sample
    size = 200 # Number of sequences to grab from the database at a time. 

    with open(DATA_DIR + write_to, 'w', encoding='utf8') as f:
        # NOTE: Not really sure why the sample isn't one-dimensional. 
        sample = np.ravel(lengths_kde.resample(size=n_lengths)) # Sample lengths from the distribution.
        for l in sample:
            # Link generated for the UniProt REST API.
            delta = 2
            lower, upper = min(0, int(l) - delta), int(l) + delta

            url = f'https://rest.uniprot.org/uniprotkb/search?format=fasta&query=%28%28length%3A%5B{lower}%20TO%20{upper}%5D%29%29&size={size}'
            # Send the query to UniProt. 
            response = requests.get(url)

            if response.status_code == 200: # Indicates query was successful.
                f.write(response.text)
            else:
                raise RuntimeError(response.text)

    # Remove duplicated, as this seems to be an issue. 
    data = fasta_to_df(read_from=write_to)
    data = data.drop_duplicates(subset=['id'])
    df_to_fasta(data, write_to=write_to)


def sum_clusters(clusters):
    '''
    Calculate the total number of entries in a set of clusters. 

    args:
        : clusters (list): A list of two-tuples where the first element is the cluster ID, and the second element is the cluster size. 
    '''
    return sum([c[1] for c in clusters] + [0])


# NOTE: Might be good to have this function mimic the one provided by sklearn. 
def train_test_split(data, test_size=0.25, train_size=0.75):
    '''
    Splits a sequence dataset into a training set and a test set. 

    args:   
        : data (pd.DataFrame): A DataFrame with, at minimum, columns 'seq', 'cluster', 'id'. 
    kwargs: 
        : test_size (float)
        : train_size (float)
    '''
    if train_size + test_size != 1.0:
        raise ValueError('Test and train sizes must sum to one.')

    # Seem to be having an issue with duplicate entries. Maybe I appended instead of overrwrote?
    # I removed all duplicates from the short.fasta file using the terminal, which hopefully fixed this anyway. 
    data = data.drop_duplicates(subset=['id']) 

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


def generate_train_and_test_data(fasta_file='all.fasta', clstr_file='all.clstr'):
    '''
    Use the cluster information and the fasta data to split the sequence data into training and test sets.  

    kwargs:
        : fasta_file (str)
        : clstr_file (str)
    '''
    clstr_data = clstr_to_df(read_from=clstr_file)
    fasta_data = fasta_to_df(read_from=fasta_file)

    # Need to combine the two DataFrames according to the ID column.
    # NOTE: inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys. 
    data = clstr_data.merge(fasta_data, how='inner', on='id')

    train_data, test_data = train_test_split(data)

    # The split seems to be working!
    # print(len(test_data) / (len(train_data) + len(test_data)))
    df_to_fasta(train_data, write_to='train.fasta')
    df_to_fasta(test_data, write_to='test.fasta')


def generate_labels(read_from='train.fasta'):
    '''
    Get labels which map each sequence in a dataset to a value indicating it is short (0) or truncated (1). 

    kwargs:
        : read_from (str)
    '''
    # Read in the sec data. Assume every non-sec protein is short. 
    sec_ids = set(fasta_to_df(read_from='sec_trunc.fasta')['id'])

    # Read in the data for which to generate the labels. 
    data = fasta_to_df(read_from=read_from)

    labels = np.zeros(len(data))
    for i in range(len(data)): # Should not be prohibitively long.
        if data['id'].iloc[i] in sec_ids:
            labels[i] = 1
    return labels


if __name__ == '__main__':
    # truncate_selenoproteins()
    # download_short_proteins()
    # generate_train_and_test_data()
    pass

