'''Code for setting up the gtdb subdirectory.'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import random
import subprocess
import os
from typing import NoReturn, Dict, List
import sys
import torch
import time
from selenobot.utils import *

GTDB_DATA_DIR = load_config_paths()['gtdb_data_dir']

def parse_taxonomy(taxonomy):
    '''Extract information from a taxonomy string.'''

    m = {'o':'order', 'd':'domain', 'p':'phylum', 'c':'class', 'f':'family', 'g':'genus', 's':'species'}

    parsed_taxonomy = {}
    # Split taxonomy string along the semicolon...
    for x in taxonomy.strip().split(';'):
        f, data = x.split('__')
        parsed_taxonomy[m[f]] = data
    
    return parsed_taxonomy # Return None if flag is not found in the taxonomy string. 


def parse_taxonomy_data(taxonomy_data):
    '''Sort the taxonomy data into columns in a DataFrame, as opposed to single strings.'''
    df = []
    for row in taxonomy_data.to_dict('records'):
        taxonomy = row.pop('taxonomy')
        row.update(parse_taxonomy(taxonomy))
        df.append(row)

    return pd.DataFrame(df)


def setup_metadata(taxonomy_files:Dict[str, str]=None, metadata_files:Dict[str, str]=None, path=None) -> NoReturn:
    '''Coalesce all taxonomy and other genome metadata information into a single CSV file.'''
    assert ('bacteria' in taxonomy_files.keys()) and ('archaea' in taxonomy_files.keys())
    assert ('bacteria' in metadata_files.keys()) and ('archaea' in metadata_files.keys())
    # First load everything into CSV files. 

    # All files should be TSV.
    dfs = []
    delimiter = '\t'
    for domain in ['bacteria', 'archaea']:
        # Read in the data, assuming it is in the same format as the originally-downloaded GTDB files. 
        taxonomy_path, metadata_path = os.path.join(GTDB_DATA_DIR, taxonomy_files[domain]), os.path.join(GTDB_DATA_DIR, metadata_files[domain])
        taxonomy_data = pd.read_csv(taxonomy_path, delimiter=delimiter, names=['genome_id', 'taxonomy'])
        metadata = pd.read_csv(metadata_path, delimiter=delimiter).rename(columns={'accession':'genome_id'}, low_memory=False)

        # Split up the taxonomy strings into separate columns for easier use. 
        taxonomy_data = parse_taxonomy_data(taxonomy_data)

        metadata = metadata.merge(taxonomy_data, on='genome_id', how='inner')
        # CheckM contamination estimate <10%, quality score, defined as completeness - 5*contamination
        metadata['checkm_quality_score'] = metadata['checkm_completeness'] - 5 * metadata['checkm_contamination']
        metadata['domain'] = domain

        dfs.append(metadata)
    
    metadata = pd.concat(dfs) # Combine the two metadata DataFrames.
    metadata.to_csv(path) # Write everything to a CSV file at the specified path.

    # Clean up the original files. 
    for file in taxonomy_files.values():
        subprocess.run(f'rm {os.path.join(GTDB_DATA_DIR, file)}', shell=True, check=True)
    for file in metadata_files.values():
        subprocess.run(f'rm {os.path.join(GTDB_DATA_DIR, file)}', shell=True, check=True)


def setup_gtdb():

    setup_metadata(
        taxonomy_files={'bacteria':os.path.join(GTDB_DATA_DIR, os)})