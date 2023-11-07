'''Utility functions for reading and writing FASTA and CSV files for the setup procedure.'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import re


def write(text, path):
    '''Writes a string of text to a file.'''
    if path is not None:
        with open(path, 'w') as f:
            f.write(text)


def read(path):
    '''Reads the information contained in a text file into a string.'''
    with open(path, 'r', encoding='UTF-8') as f:
        text = f.read()
    return text


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


def csv_size(path):
    '''Get the number of entries in a FASTA file.'''
    return len(csv_ids(path))


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
    dfs = [pd_from_fasta(p, set_index=False) for p in paths]
    df = pd.concat(dfs)
    
    # Remove any duplicates following concatenation. 
    n = len(df)
    df = df.drop_duplicates(subset='id')
    df = df.set_index('id')

    if len(df) < n:
        print(f'utils.fasta_concatenate: {n - len(df)} duplicates removed upon concatenation.')

    pd_to_fasta(df, path=out_path)


def pd_from_fasta(path, set_index=False):
    '''Load a FASTA file in as a pandas DataFrame.'''

    ids = fasta_ids(path)
    seqs = fasta_seqs(path)

    df = pd.DataFrame({'seq':seqs, 'id':ids})
    # df = df.astype({'id':str, 'seq':str})
    if set_index: 
        df = df.set_index('id')
    return df


def pd_to_fasta(df, path=None, textwidth=80):
    '''Convert a pandas DataFrame containing FASTA data to a FASTA file format.'''

    assert df.index.name == 'id', 'setup.pd_to_fasta: Gene ID must be set as the DataFrame index before writing.'

    fasta = ''
    for row in tqdm(df.itertuples(), desc='utils.df_to_fasta', total=len(df)):
        fasta += '>|' + str(row.Index) + '|\n'

        # Split the sequence up into shorter, sixty-character strings.
        n = len(row.seq)
        seq = [row.seq[i:min(n, i + textwidth)] for i in range(0, n, textwidth)]

        seq = '\n'.join(seq) + '\n'
        fasta += seq
    
    # Write the FASTA string to the path-specified file. 
    write(fasta, path=path)


def pd_from_clstr(clstr_file_path):
    '''Convert a .clstr file string to a pandas DataFrame. The resulting DataFrame maps cluster label to gene ID.'''
    # Read in the cluster file as a string. 
    clstr = read(clstr_file_path)
    df = {'id':[], 'cluster':[]}
    # The start of each new cluster is marked with a line like ">Cluster [num]"
    clusters = re.split(r'^>.*', clstr, flags=re.MULTILINE)
    # Split on the newline. 
    for i, cluster in enumerate(clusters):
        ids = [get_id(x) for x in cluster.split('\n') if x != '']
        df['id'] += ids
        df['cluster'] += [i] * len(ids)

    df = pd.DataFrame(df) # .set_index('id')
    df.cluster = df.cluster.astype(int) # This will speed up grouping clusters later on. 
    return df


def fasta_ids_with_min_seq_length(fasta_file_path, min_seq_length=6):
    '''A function for grabbing all gene IDs in a FASTA file for which the corresponding sequences meet the minimum
    length requirement specified in the setup.py file.'''
    ids, seqs = fasta_ids(fasta_file_path), fasta_seqs(fasta_file_path)
    seq_lengths = np.array([len(s) for s in seqs]) 
    # Filter IDs which do not meet the minimum sequence length requirement. 
    return ids[seq_lengths >= min_seq_length]


def fasta_seqs_with_min_seq_length(fasta_file_path, min_seq_length=6):
    '''A function for grabbing all gene IDs in a FASTA file for which the corresponding sequences meet the minimum
    length requirement specified in the setup.py file.'''
    seqs = fasta_seqs(fasta_file_path)
    return [s for s in seqs if len(s) >= min_seq_length]


def fasta_size_with_min_seq_length(fasta_file_path, min_seq_length=6):
    '''Get the number of sequenes in a FASTA file which meet the minimum sequence lengh requirement.'''
    seq_lengths = np.array([len(s) for s in fasta_seqs(fasta_file_path)])
    return np.sum(seq_lengths >= min_seq_length)


def pd_from_fasta_with_min_seq_length(path, set_index=False, min_seq_length=6):
    '''Load a FASTA file in as a pandas DataFrame.'''

    ids = fasta_ids_with_min_seq_length(path, min_seq_length=min_seq_length)
    seqs = fasta_seqs_with_min_seq_length(path, min_seq_length=min_seq_length)

    df = pd.DataFrame({'seq':seqs, 'id':ids})
    # df = df.astype({'id':str, 'seq':str})
    if set_index: 
        df = df.set_index('id')
    return df



