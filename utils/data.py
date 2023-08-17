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


def fasta_to_df(fasta):
    '''Read data from a FASTA file and convert it to a pandas DataFrame. The 
    resulting DataFrame contains a column for the sequence and a column for 
    the unique identifier.
    '''
    ids = [get_id(head) for head in re.findall(r'^>.*', fasta, re.MULTILINE)]
    seqs = re.split(r'^>.*', fasta, flags=re.MULTILINE)[1:]
    # Strip all of the newline characters from the amino acid sequences. 
    seqs = [s.replace('\n', '') for s in seqs]
    
    df = pd.DataFrame({'seq':seqs, 'id':ids})
    df = df.astype({'id':'str', 'seq':'str'})
    df = df.set_index('id')

    return df


def df_to_fasta(df, path=None, textwidth=80):
    '''Convert a DataFrame containing FASTA data to a FASTA file format.'''

    fasta = ''
    for row in df.itertuples():
        fasta += '>  |' + row.id + '|\n'
        # Split the sequence up into shorter, sixty-character strings.
        n = len(row.seq)
        seq = [row.seq[i:min(n, i + textwidth)] for i in range(0, n, textwidth)]

        assert ''.join(seq) == row.seq # Make sure no information was lost. 

        seq = '\n'.join(seq) + '\n'
        fasta += seq
    
    # Write the FASTA string to the path-specified file. 
    write(fasta, path=path)

    return fasta


def clstr_to_df(clstr_file):
    '''
    Reads a clustr file (the output file from CD-hit) into a pandas DataFrame. 
    DataFrame has columns for the cluster number and sequence ID (it discards 
    a lot of the info in the clstr file). 

    kwargs:
        - clstr_file (str): The path to the file containing the cluster data. 

    returns: pd.DataFrame
    '''
    cluster_count = 0
    df = {'cluster':[], 'id':[]}

    with open(clstr_file, 'r', encoding='utf8') as f:
        lines = f.readlines()
        i = 1 # Skip the first line, which is just a cluster header. 

        while i < len(lines):
            line = lines[i]
            if 'Cluster' not in line:
                line = line.split() # Split along whitespace.
                
                df['cluster'].append(cluster_count)
                df['id'].append(get_id(line[2]))

            else: # Move on to the next cluster. 
                cluster_count += 1

            i += 1 # Increment the line number, regardless of whether or not the cluster count is incremented. 

    # Convert to a DataFrame and return. Make sure datatypes are correct.  
    return pd.DataFrame(df).astype({'cluster':'int64', 'id':'str'})


def truncate_selenoproteins(fasta, path=None):
    '''Truncate the selenoproteins stored in the inputted file at the first 
    selenocysteine residue. Overwrites the original file with the truncated 
    sequence data.'''
    df = fasta_to_df(fasta_file)

    df_trunc = {'id':[], 'seq':[]}
    for row in df.itertuples():
        
        # Find indices where a selenocysteine occurs. 
        idxs = np.where(list(row.seq) == 'U')
        # Sequentially truncate at each selenocysteine redidue. 
        seqs = [row.seq[:idx] for idx in idxs]
        # Add new truncated sequences, and the corresponding gene IDs, to the
        # dictionary which will become the new DataFrame. 
        df_trunc['id'] += [row.id] * len(seqs)
        df_trunc['seq'] += seqs

    df_trunc = pd.DataFrame(df_trunc)
    fasta = df_to_fasta(df_trunc, path=path)

    return fasta # Return a FASTA file with the truncated proteins. 


def sum_clusters(clusters):
    '''
    Calculate the total number of entries in a set of clusters. 

    args:
        - clusters (list): A list of two-tuples where the first element is the 
            cluster ID, and the second element is the cluster size. 
    '''
    return sum([c[1] for c in clusters] + [0])


def train_test_split(data, test_size=0.25, train_size=0.75):
    '''
    Splits a sequence dataset into a training set and a test set. 

    args:   
        - data (pd.DataFrame): A DataFrame with, at minimum, columns 'seq', 'cluster', 'id'. 
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
    # The cluster size information isn't relevant anymore, so remove it.
    train_clusters = [c[0] for c in train_clusters]
    test_clusters = [c[0] for c in test_clusters]

    test_data = data[data['cluster'].isin(test_clusters)]
    train_data = data[data['cluster'].isin(train_clusters)]
    
    return train_data, test_data # For now, just return the full DataFrames.



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


def get_sec_metadata_from_uniprot(fields=['lineage'], path=None):
    '''Download metadata from UniProt in TSV format using their API. By default,
    downloads the taxonomic lineage, as well as the eggNOG orthology group.'''

    fields = '%2C'.join(fields)
    url = f'https://rest.uniprot.org/uniprotkb/stream?compressed=false&fields=accession%2C{fields}&format=tsv&query=%28%28ft_non_std%3Aselenocysteine%29%29'
    tsv = requests.get(url).text

    if 'lineage' in fields:
        # Clean up the metadata by removing all non-domain taxonomic information.
        func = lambda l : l.split(',')[1].strip().split(' ')[0].lower()
        df = pd.read_csv(io.StringIO(tsv), delimiter='\t')
        df['domain'] = df['Taxonomic lineage'].apply(func)
        df = df.drop(columns=['Taxonomic lineage'])
        df = df.rename(columns={'Entry':'id', 'eggNOG':'cog'})
        df = df.set_index('id')

        # Convert back into a text string. 
        tsv = df.to_csv(sep='\t')
        write(tsv, path=path)

    return tsv


def get_sec_from_uniprot(path=None):
    '''Uses the UniProt API to download all known selenoproteins from the database.
    It returns a text string with the information in FASTA format.'''

    url = 'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28%28ft_non_std%3Aselenocysteine%29%29'
    fasta = requests.get(url).text

    write(fasta, path)

    return fasta


def main(data_dir='/home/prichter/Documents/selenobot/data/'):
    

    today = date.today().strftime('%m%d%y')

    # get_sec_from_uniprot(path=data_dir + f'uniprot_{today}_sec.fasta'
    # get_sec_metadata_from_uniprot(path=data_dir + f'uniprot_{today}_sec_metadata.tsv')
    pass

if __name__ == '__main__':
    main()

