'''
Code for gathering and preprocessing the data in the /protex/data directory. 
'''

# TODO: Replicate procedure I used to generate all the data files in thie file, mostly
# for the sake of record-keeping. 

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import time
from transformers import EsmTokenizer, AutoTokenizer, EsmModel


def get_id(metadata):
    '''
    Extract the unique identifier from a FASTA metadata string (the 
    information on the line preceding the actual sequence). This ID should 
    be flanked by '|'.

    args:
        - metadata (str): A metadata string containing an ID. 
    '''
    start_idx = metadata.find('|') + 1
    # Cut off any extra stuff preceding the ID, and locate the remaining |.
    metadata = metadata[start_idx:]
    end_idx = metadata.find('|')

    return str(metadata[:end_idx])


def fasta_to_df(fasta_file):
    '''
    Read data from a FASTA file and convert it to a pandas DataFrame. The 
    resulting DataFrame contains a column for the sequence and a column for 
    the unique identifier. 

    args:
        - fasta_file (str): Either a path to the FASTA file or the contents of the file. 
    '''
    df = {'seq':[], 'id':[], 'metadata':[]}

    try:
    # Open the FASTA file, or read in the string.  
        with open(fasta_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
    except:
        lines = fasta_file.split('\n')
        lines = [line for line in lines if len(line) > 0]
    
    # Metadata is contained on a single line, with the sequence given on the subsequent lines.
    # The sequence is broken up across multiple lines, with each line containing 60 characters.
    i = 0 
    while i < len(lines):
        # Also extract the unique ID from the metadata line, and add to the DataFrame. 
        df['metadata'].append(lines[i])
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


def df_to_fasta(df, fasta_file):
    '''
    Write a DataFrame containing FASTA data to a FASTA file format.

    args:
        - df (pd.DataFrame): A DataFrame containing, at minimum, a metadata column, 
            as well as a sequence column. 
        - fasta_file (str): The path to the file which will contain the FASTA data. 
    '''
    with open(fasta_file, 'w', encoding='utf8') as f:
        # Iterate over the DataFrames as a set of named tuples.
        for row in df.itertuples():
            f.write(row.metadata)
            
            # Split the sequence up into shorter, sixty-character strings.
            n = len(row.seq)
            seq = [row.seq[i:min(n, i + 60)] for i in range(0, n, 60)]
            seq = '\n'.join(seq) + '\n'
            f.write(seq)


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


def truncate(seq):
    '''
    Truncate the input amino acid sequence at the first selenocysteine residue,
    which is denoted "U."
    '''
    idx = seq.find('U')
    return seq[:idx]


def truncate_selenoproteins(fasta_file):
    '''
    Truncate the selenoproteins stored in the inputted file at the first 
    selenocysteine residue. Overwrites the original file with the truncated 
    sequence data. 

    args:
        - fasta_file (str): The path to the file which contains FASTA data. 
    '''
    data = fasta_to_df(fasta_file)

    # Truncate the data at the first 'U' residue
    func = np.vectorize(truncate, otypes=[str])
    data['seq'] = data['seq'].apply(func)

    # Write the DataFrame in FASTA format. THIS OVERWRITES THE FILE.
    df_to_fasta(data, fasta_file)


def download_short_proteins(dist, fasta_file):
    '''
    Download short proteins from the UniProt database according to the
    length distribution of the truncated selenoproteins.

    args:
        - dist (?): A distribution generated using a kernel density estimate. 
        - fasta_file (str): The path to the file which contains FASTA data. 
    '''
    delta = 0
    # There are about 20,000 selenoproteins listed, so try to get an equivalent number of short proteins. 
    n_lengths = 200 # The number of lengths to sample
    
    # Dictionary mapping the length to the link to the next page. 
    history = {}

    with open(fasta_file, 'w', encoding='utf8') as f:
        # NOTE: Not really sure why the sample isn't one-dimensional. 
        sample = np.ravel(dist.resample(size=n_lengths)) # Sample lengths from the distribution.
        sample = np.asarray(sample, dtype='int') # Convert to integers. 

        for l in tqdm(sample, desc='Downloading protein sequences...'):
            
            if l in history: # For pagination. 
                if history[l] is None:
                    continue
                # Clean up the URL string stored from the Link header. 
                url = history[l].split()[0].replace('<', '').replace('>', '').replace(';', '')
            
            else: # Link generated for the UniProt REST API.
                url = f'https://rest.uniprot.org/uniprotkb/search?format=fasta&query=%28%28length%3A%5B{l}%20TO%20{l}%5D%29%29&size=100'
            
            # Send the query to UniProt. 
            try:
                response = requests.get(url)
                # Sometimes a link isn't returned? Maybe means no more results?
                history[l] = response.headers.get('Link', None)

                if response.status_code == 200: # Indicates query was successful.
                    f.write(response.text)
                else:
                    msg = f'The URL {f} threw the following error:\n\n{response.text}'
                    raise RuntimeError(msg)
            except requests.packages.urllib3.exceptions.ProtocolError:
                print('UniProt thinks it is being scraped. Wait a couple of minutes befoe continuing.')
                time.sleep(60)


    # Remove duplicates, as this seems to be an issue. 
    data = fasta_to_df(fasta_file)
    data = data.drop_duplicates(subset=['id'])
    df_to_fasta(data, fasta_file)


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


def generate_esm_embeddings(data, embedding_file=None):
    '''
    Run a file full of amino acid sequences through the ESM model in order to generate
    embeddings. 

    args: 
        - data (pd.DataFrame)
        - filename (str)
    '''
    name = 'facebook/esm2_t6_8M_UR50D' # Name of the pre-trained model. 
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = EsmModel.from_pretrained(name)

    inputs = tokenizer(data['seqs'])

    

if __name__ == '__main__':
    pass
    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------

# def generate_labels(data, sec_data):
#     '''
#     Get labels which map each sequence in a dataset to a value indicating it is short (0) or truncated (1). 
# 
#     args:
#         - data (pd.DataFrame): The sequence data, which must at least contain an 'id' column. 
#         - sec_data (pd.DataFrame): The data containing truncated selenoprotein sequences. 
# 
#     returns: torch.Tensor
#     '''
#     # Get the IDs for all selenoproteins from the selenoprotein data.  
#     sec_ids = list(sec_data['id'])
# 
#     labels = np.zeros(len(data), dtype=np.single) # Specify integer type. 
#     for i in range(len(data)): # Should not be prohibitively long.
#         if data['id'].iloc[i] in sec_ids:
#             labels[i] = 1
# 
#     # Should have shape (batch_size, )
#     labels = torch.from_numpy(labels) # Convert to tensor.
#     labels = torch.unsqueeze(labels, 1) # Hopefully this fixes the dimension issue. 
#     return labels # .long() # Make sure labels are integers. 






