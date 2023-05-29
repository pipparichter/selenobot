'''
'''

# Silence all the deprecation warnings being triggered by umap-learn.
import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings('ignore') 

import pandas as pd
import numpy as np
import torch
import tensorflow as tf


def get_id(metadata):
    '''
    Extract the unique identifier from a FASTA metadata string (the information on the line preceding
    the actual sequence). This ID should be flanked by '|'.

    args:
        - metadata (str): A metadata string containing an ID. 

    returns: str
    '''
    start_idx = metadata.find('|') + 1
    # Cut off any extra stuff preceding the ID, and locate the remaining |.
    metadata = metadata[start_idx:]
    end_idx = metadata.find('|')

    return str(metadata[:end_idx])


def fasta_to_df(fasta_file):
    '''
    Read data from a FASTA file and convert it to a pandas DataFrame. The resulting DataFrame contains
    a column for the sequence and a column for the unique identifier. 

    args:
        - fasta_file (str)
    
    returns: pd.DataFrame
    '''
    df = {'seq':[], 'id':[], 'metadata':[]}

    # Open the FASTA file. Metadata is contained on a single line, with the sequence given on the subsequent lines.
    # The sequence is broken up across multiple lines, with each line containing 60 characters.
    with open(fasta_file, 'r', encoding='utf8') as f:
        lines = f.readlines()

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
        - df (pd.DataFrame): A DataFrame containing, at minimum, a metadata column, as well as a sequence column. 
        - fasta_file (str): The path to the file. 
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
    Reads a clustr file (the output file from CD-hit) into a pandas DataFrame. DataFrame has columns
    for the cluster number and sequence ID (it discards a lot of the info in the clstr file). 

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


