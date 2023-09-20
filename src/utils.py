'''A set of miscellaneous utility functions to make working with FASTA and CSV data easier.'''
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


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


def fasta_concatenate(paths, out_path=None, verbose=True):
    '''Combine the FASTA files specified by the paths. Creates a new file
    containing the combined data.'''
    dfs = [pd_from_fasta(p, set_index=False) for p in paths]
    df = pd.concat(dfs)
    
    # Remove any duplicates following concatenation. 
    n = len(df)
    df = df.drop_duplicates(subset='id')
    df = df.set_index('id')

    if len(df) < n and verbose: print(f'utils.fasta_concatenate: {n - len(df)} duplicates removed upon concatenation.')

    pd_to_fasta(df, path=out_path)


# def fasta_check(path, verbose=True):
#     '''Confirm that there are no duplicate entries (or anything else wrong) in the FASTA file.'''
#     filename = os.path.split(path)[-1] # Extrac thte filename for debugging purposes. 

#     ids = fasta_ids(path)
#     assert len(np.unique(ids)) == len(ids), f'utils.fasta_check: Duplicate IDs are present in {filename}.'

#     seqs = fasta_seqs(path)

#     unlabeled_ids = [id_[:-3] if '[1]' in id_ else id_ for id_ in ids]
#     assert len(np.unique(unlabeled_ids)) == len(ids), 'utils.fasta_check: Selenoproteins are present in both truncated and full-length forms in {filename}.'
            

def csv_concatenate(paths, out_path=None):
    '''Combine the CSV files specified by the paths. Creates a new file
    containing the combined data.'''

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs)
    n = len(df)

    # Remove any duplicates following concatenation. 
    df = df.drop_duplicates(subset='id')
    df = df.set_index('id')
    
    if len(df) < n and verbose: print(f'utils.csv_concatenate: {n - len(df)} duplicates removed upon concatenation.')
    
    df.to_csv(out_path)


def pd_from_fasta(path, set_index=True):
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

    assert df.index.name == 'id', 'Gene ID must be set as the DataFrame index before writing.'

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
