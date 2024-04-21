'''Utility functions for reading and writing FASTA and CSV files, amongst other things.'''
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
from typing import Dict, NoReturn
import configparser
import pickle
import subprocess

# Define some important directories...
CWD, _ = os.path.split(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CWD, '..', 'results') # Get the path where results are stored.
DATA_DIR = os.path.join(CWD, '..', 'data') # Get the path where results are stored. 

def to_numeric(n:str):
    '''Try to convert a string to a numerical data type. Used when 
    reading in header strings from files.'''
    try: 
        n = int(n)
    except:
        try: 
            n = float(n)
        except:
            pass
    return n


def write(text:str, path:str) -> NoReturn:
    '''Writes a string of text to a file.   
    
    :param text: The text to write to the file.
    :param path: The path of the file to which the text will be written. 
    '''
    if path is not None:
        with open(path, 'w') as f:
            f.write(text)


def read(path:str) -> str:
    '''Reads the information contained in a text file into a string.
    
    :param path: The path to the file to read. 
    :return: The text contained in the file as a Python string.
    '''
    with open(path, 'r') as f: #, encoding='UTF-8') as f:
        text = f.read()
    return text


def fasta_ids(path:str, id_label:str='id') -> np.array:
    '''Extract all gene IDs stored in a FASTA file.'''
    # Read in the FASTA file as a string. 
    fasta = read(path)
    # Extract all the IDs from the headers, and return the result. 
    pattern = f'{id_label}=([^;]+)'
    ids = [re.search(pattern, head).group(1) for head in re.findall(r'^>.*', fasta, re.MULTILINE)]
    return np.array(ids)


def fasta_size(path:str) -> int:
    '''Gets the number of entries in a FASTA file.'''
    return len(fasta_ids(path))


def csv_size(path):
    '''Get the number of entries in a CSV file.'''
    n = subprocess.run(f'wc -l {path}', capture_output=True, text=True, shell=True, check=True).stdout.split()[0]
    n = int(n) - 1 # Convert text output to an integer and disregard the header row.
    return n


def fasta_seqs(path):
    '''Extract all amino acid sequences stored in a FASTA file.'''
    # Read in the FASTA file as a string. 
    fasta = read(path)
    seqs = re.split(r'^>.*', fasta, flags=re.MULTILINE)[1:]
    # Strip all of the newline characters from the amino acid sequences. 
    seqs = [s.replace('\n', '') for s in seqs]
    # return np.array(seqs)
    return seqs
    

def dataframe_from_fasta(path:str, parse_header:bool=True) -> pd.DataFrame:
    '''Load the database FASTA file in as a pandas DataFrame.'''
    df = {'seq':[]}
    text = read(path)

    seqs = re.split(r'^>.*', text, flags=re.MULTILINE)[1:]
    # Strip all of the newline characters from the amino acid sequences. 
    seqs = [s.replace('\n', '') for s in seqs]
    headers = re.findall(r'^>.*', text, re.MULTILINE)

    for seq, header in zip(seqs, headers):
        # Headers are of the form |>col=value|...|col=value
        header = header.replace('>', '') # Remove the header marker. 
        header = [entry.split('=') for entry in header.split(';')] if parse_header else [('id', header)]
        for col, val in header:
            if col not in df:
                df[col] = []
            df[col].append(val)
        df['seq'].append(seq) # Add the sequence as well. 

    df = pd.DataFrame(df) # Convert to a DataFrame

    # Convert to numerical datatypes. 
    num_fields = ['aa_length', 'nt_start', 'nt_stop', 'nt_ext']
    for field in num_fields:
        if field in df.columns:
            df[field] = df[field].apply(pd.to_numeric)
    if 'reverse' in df.columns:
        df['reverse'] = df['reverse'].apply(bool)

    return df


def dataframe_to_fasta(df:pd.DataFrame, path:str, textwidth:int=80) -> NoReturn:
    '''Convert a pandas DataFrame containing FASTA data to a FASTA file format.

    :param df: The pandas DataFrame containing, at minimum, a column of sequences and
        a column of gene IDs.
    :param path: The path to write the FASTA file to. 
    :param textwidth: The length of lines in the FASTA file.     
    '''
    # Sometimes the ID column is the 
    if df.index.name in ['gene_id', 'id']:
        df[df.index.name] = df.index

    # Include all non-sequence fields in the FASTA header. 
    header_fields = [col for col in df.columns if col != 'seq']

    fasta = ''
    for row in df.itertuples():
        header = [f'{field}={getattr(row, field)}' for field in header_fields]
        fasta += '>' + ';'.join(header) + '\n' # Add the header to the FASTA file. 

        # Split the sequence up into substrings of length textwidth.
        n = len(row.seq)
        seq = [row.seq[i:min(n, i + textwidth)] for i in range(0, n, textwidth)]
        assert len(''.join(seq)) == n, 'utils.pd_to_fasta: Part of the sequence was lost when splitting into lines.'
        fasta += '\n'.join(seq) + '\n'

    # Write the FASTA string to the path-specified file. 
    write(fasta, path=path)


def dataframe_from_gff(path:str, cds_only:bool=True) -> pd.DataFrame:
    '''convert a .gff gile into a pandas DataFrame.

    :param path: The path to the .gff file.
    :param cds_only: Whether or not to only include coding sequences. True by default. 
    :return: A pandas DataFrame containing information in the GFF file.
    '''
    # GFF files are tab-separated. There are no column headers, but the fields are as follows.
    cols = ['scaffold_id', 'feature', 'nt_start', 'nt_stop', 'score', 'reverse', 'frame', 'info']

    # id: The name of the chromosome or scaffold (must not contain assembly information)
    # source: The name of the program that generated this feature
    # feature: Feature type name, e.g. Gene, Variation, Similarity
    # nt_start: Start position of the feature, with sequence numbering starting at 1.
    # nt_stop: End position of the feature, with sequence numbering starting at 1.
    # score: A floating point value.
    # strand: + (forward) or - (reverse).
    # frame: One of '0', '1' or '2'. '0' indicates that the first base of the feature is the first base of a codon, '1' that the second base is the first base of a codon, and so on.
    # attribute: A semicolon-separated list of tag-value pairs, providing additional information about each feature.

    # Skip the comment lines at the beginning using the comment parameter. 
    df = pd.read_csv(path, delimiter='\t', names=cols, comment='#') # Load in the TSV with the correct column names. 
    
    def parse_info(infos:pd.Series) -> pd.DataFrame():
        '''Parse the info strings, extracting the relevant information. This function
        assumes all non-coding sequences have been filtered out of the DataFrame.'''
        info_df = {'id':[], 'gene_id':[]}
        for info in infos:
            # Extract the GenBank gene identifier. 
            id_match = re.search('ID=cds-([^;]+);', info)
            info_df['id'].append(id_match.group(1))
            # Extract the gene name. 
            gene_id_match = re.search(f'gene=([^;]+);', info)
            if gene_id_match is None:
                info_df['gene_id'].append(None)
            else:
                info_df['gene_id'].append(gene_id_match.group(1))
        # Drop the source column, as it can contains whitespaces that trip up MMSeqs2 (causes it to cut off some of the header when 
        # writing the results file).
        return pd.DataFrame(info_df)

    # Clean up the data a bit... 
    df['reverse'] = [x == '-' for x in df.reverse if x != '.'] # Convert to booleans. 
    #  Both the start and end position are inclusive and one-indexed.
    df['nt_start'] = df.nt_start - 1 # Shift to zero-indexed.
    # Not adjusting the upper bound means it can be used directly in a slice. 
    # df['nt_stop'] = df.nt_stop - 1 # Shift to zero-indexed.

    # Fill blanks with NaNs. 
    df = df.replace('.', np.nan)

    # Filter out everything that's not a coding sequence. Make sure to reset the index so pd.concat works correctly.
    df = df[df['feature'].str.match('CDS')].reset_index(drop=True)
    df['frame'] = df.frame.astype(int)
    df = pd.concat([df, parse_info(df['info'])], axis=1)

    return df.drop(columns=['info', 'score']).reset_index(drop=True) # Drop some columns we don't care about. 


def dataframe_from_clstr(path:str) -> pd.DataFrame:
    '''Convert a .clstr file string to a pandas DataFrame. 
    
    :param path: The path to the .clstr file generated by CD-HIT.
    :return: A pandas DataFrame mapping gene ID to cluster ID. 
    '''
    clstr = read(path) # Read in the cluster file as a string. 
    df = {'id':[], 'cluster':[]}
    # The start of each new cluster is marked with a line like ">Cluster [num]"
    clusters = re.split(r'^>.*', clstr, flags=re.MULTILINE)
    # Split on the newline. 
    for i, cluster in enumerate(clusters):
        pattern = '>id=([\w\d_\[\]]+)' # Pattern to extract gene ID from line. 
        ids = [re.search(pattern, x).group(1) for x in cluster.split('\n') if x != '']
        df['id'] += ids
        df['cluster'] += [i] * len(ids)

    df = pd.DataFrame(df) # .set_index('id')
    df.cluster = df.cluster.astype(int) # This will speed up grouping clusters later on. 
    return df


def dataframe_from_m8(path:str) -> pd.DataFrame:
    '''Load a TSV file produced by running BLAST pairwise alignment.'''
    # Column names for output format 6. From https://www.metagenomics.wiki/tools/blast/blastn-output-format-6. 
    columns =['query_header', 'target_header', 'percentage_identical', 'align_length', 'num_mismatches', 'num_gap_openings', 
        'query_align_start', 'query_align_stop', 'target_align_start', 'target_align_stop', 'e_value', 'bit_score', 'query_align_seq', 'target_align_seq']
    # Read in the TSV file. 
    df = pd.read_csv(path, index_col=None, delimiter='\t', names=columns)

    def parse_headers(headers:pd.Series, prefix:str) -> pd.DataFrame:
        '''Convert a series of headers into a pandas DataFrame.'''
        rows = []
        for header in headers:
            try:
                # Assume the header is separated by semicolons, as for other FASTA files in this project. 
                header = dict([item.split('=') for item in header.split(';')])
            except: # For the MMSeqs2-generated files, the target_headers only contain the GTDB gene ID. 
                header = {'id':header}
            items = list(header.items())
            for key, val in items: # Rename the header fields to indicate query or target.
                del header[key] # Remove the old key from the header. 
                header[f'{prefix}_{key}'] = to_numeric(val)
            rows.append(header)
        return pd.DataFrame(rows, index=np.arange(len(headers)))

    headers = pd.concat([parse_headers(df.target_header, 'target'), parse_headers(df.query_header, 'query')], axis=1)
    df = df.drop(columns=['target_header', 'query_header'])
    df = pd.concat([df, headers], axis=1)

    # data['u_pos'] = data.query_align_seq.str.find('U')
    # data['u_overlap'] = data.apply(lambda row: row.target_align_seq[row.u_pos], axis=1)

    return df







