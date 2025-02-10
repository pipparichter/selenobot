'''Assorted utility functions which are useful in multiple scripts and modules.'''
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
from typing import Dict, NoReturn, List, Tuple
import configparser
import pickle
import subprocess
import json
import random
from collections import OrderedDict
import torch


GTDB_DTYPES = dict()
GTDB_DTYPES['description'] = str
GTDB_DTYPES['seq'] = str
GTDB_DTYPES['start'] = int
GTDB_DTYPES['stop'] = int
GTDB_DTYPES['strand'] = int
GTDB_DTYPES['ID'] = str
GTDB_DTYPES['partial'] = str
GTDB_DTYPES['start_type'] = str
GTDB_DTYPES['rbs_motif'] = str
GTDB_DTYPES['rbs_spacer'] = str
GTDB_DTYPES['gc_content'] = float
GTDB_DTYPES['genome_id'] = str
GTDB_DTYPES['ambiguous_bases'] = int
GTDB_DTYPES['checkm_completeness'] = float
GTDB_DTYPES['checkm_contamination'] = float
GTDB_DTYPES['checkm_marker_count'] = int
GTDB_DTYPES['checkm_marker_lineage'] = str
GTDB_DTYPES['checkm_marker_set_count'] = int
GTDB_DTYPES['coding_bases'] = int
GTDB_DTYPES['coding_density'] = float
GTDB_DTYPES['contig_count'] = int
GTDB_DTYPES['gc_count'] = int
GTDB_DTYPES['gc_percentage'] = float
GTDB_DTYPES['genome_size'] = int
GTDB_DTYPES['gtdb_genome_representative'] = int
GTDB_DTYPES['gtdb_representative'] = int
GTDB_DTYPES['gtdb_taxonomy'] = int
GTDB_DTYPES['gtdb_type_species_of_genus'] = int
GTDB_DTYPES['l50_contigs'] = int
GTDB_DTYPES['l50_scaffolds'] = int
GTDB_DTYPES['longest_contig'] = int
GTDB_DTYPES['longest_scaffold'] = int
GTDB_DTYPES['mean_contig_length'] = int
GTDB_DTYPES['mean_scaffold_length'] = int
GTDB_DTYPES['mimag_high_quality'] = str
GTDB_DTYPES['mimag_low_quality'] = str
GTDB_DTYPES['mimag_medium_quality'] = str
GTDB_DTYPES['n50_contigs'] = int
GTDB_DTYPES['n50_scaffolds'] = int
GTDB_DTYPES['ncbi_assembly_level'] = str
GTDB_DTYPES['ncbi_bioproject'] = str
GTDB_DTYPES['ncbi_biosample'] = str
GTDB_DTYPES['ncbi_genbank_assembly_accession'] = str
GTDB_DTYPES['ncbi_genome_category'] = str
GTDB_DTYPES['ncbi_refseq_category'] = str
GTDB_DTYPES['ncbi_genome_representation'] = str
GTDB_DTYPES['ncbi_isolate'] = str
GTDB_DTYPES['ncbi_isolation_source'] = str
GTDB_DTYPES['ncbi_translation_table'] = int
GTDB_DTYPES['ncbi_scaffold_count'] = int
GTDB_DTYPES['ncbi_species_taxid'] = int
GTDB_DTYPES['ncbi_taxid'] = int
GTDB_DTYPES['ncbi_taxonomy'] = str
GTDB_DTYPES['protein_count'] = int
GTDB_DTYPES['scaffold_count'] = int
GTDB_DTYPES['total_gap_length'] = int
GTDB_DTYPES['trna_aa_count'] = int
GTDB_DTYPES['trna_count'] = int
GTDB_DTYPES['trna_selenocysteine_count'] = int
GTDB_DTYPES['phylum'] = str
GTDB_DTYPES['class'] = str
GTDB_DTYPES['order'] = str
GTDB_DTYPES['genus'] = str
GTDB_DTYPES['species'] = str
GTDB_DTYPES['family'] = str
GTDB_DTYPES['domain'] = str
GTDB_DTYPES['prefix'] = str


def apply_gtdb_dtypes(df:pd.DataFrame):
    dtypes = {col:dtype for col, dtype in GTDB_DTYPES.items() if (col in df.columns)}
    return df.astype(dtypes)


def load_gtdb_genome_metadata(path:str, reps_only:bool=True):
    '''Load in a GTDB metadata TSV file and fix the columns to make them more usable.'''
    df = pd.read_csv(path, delimiter='\t', low_memory=False, dtype={'partial':str})
    df = df.rename(columns={'accession':'genome_id'})
    df['prefix'] = [genome_id[:2] for genome_id in df.genome_id] 
    df['genome_id'] = [genome_id.replace('GB_', '').replace('RS_', '') for genome_id in df.genome_id] # Remove the prefixes from the genome IDs.
    df = df[df.gtdb_representative == 't'] # Get the representatives. 
    
    for level in ['phylum', 'class', 'order', 'genus', 'species', 'family', 'domain']: # Parse the taxonomy string. 
        df[level] = [re.search(f'{level[0]}__([^;]*)', taxonomy).group(1) for taxonomy in df.gtdb_taxonomy]

    drop = [col for col in df.columns if col not in GTDB_DTYPES.keys()]
    print(f'load_genome_metadata: Dropping {len(drop)} columns from the DataFrame.')
    df = df.drop(columns=drop)

    for col, dtype in GTDB_DTYPES.items():
        if col in df.columns:
            if dtype in [int, float]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if dtype == [str]:
                df[col] = df[col].astype(str)

    return df.set_index('genome_id')


def seed(seed:int=42) -> None:
    '''Seed all random number generators I can think of for the sake of reproducibility.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # # Set a fixed value for the hash seed (not sure what this does, so got rid of it)
    # os.environ["PYTHONHASHSEED"] = str(seed)

def trunc_n_terminus(seq:str, min_length:int=10, allowed_starts:list=['M', 'V', 'L']):
    '''Truncate a selenoprotein at the N-terminal end.
    
    :param seq
    :param min_length
    :param allowed_starts
    '''
    # Default allowed_starts are amino acids coded for by the traditional start and alternative start codons. 
    # Methionine is coded by AUG, Valine by GUG, and Leucine by UUG. Frequencies here: https://pmc.ncbi.nlm.nih.gov/articles/PMC5397182/ 
    idx = seq.rindex('U') # Get the index of the rightmost selenocysteine. 
    seq = seq[idx + 1:]
    for aa in allowed_starts:
        idx = seq.find(aa)
        if (idx >= 0) and (len(seq[idx:]) >= min_length):
            return seq[idx:]
    return None


def trunc_c_terminus(seq:str, min_length:int=10):
    '''Truncate a selenoprotein at the C-terminal end.'''
    idx = seq.index('U') # Get the index of the leftmost selenocysteine. 
    if len(seq[:idx]) >= min_length:
        return seq[:idx]
    else:
        return None


def truncate_sec(metadata_df:pd.DataFrame, terminus:str='c', min_length:int=10, drop_failures:bool=True, **kwargs) -> pd.DataFrame:
    '''Truncate the selenoproteins in the DataFrame.'''
    metadata_trunc_df = []
    trunc_func = trunc_c_terminus if (terminus == 'c') else trunc_n_terminus

    n_failures = 0

    ids = list()
    for row in tqdm(metadata_df.to_dict(orient='records', index=True), 'truncate_sec: Truncating selenoproteins...'):
         
        seq = row['seq'] # Extract the sequence from the row.
        
        row['sec_index_n'] = seq.index('U') 
        row['sec_index_c'] = seq.rindex('U') 
        row['sec_count'] = seq.count('U') # Store the number of selenoproteins in the original sequence.

        seq_trunc = trunc_func(seq, min_length=min_length, **kwargs)
        if seq_trunc is None:
            n_failures += 1
        else:
            row['trunc_size'] = len(seq) - len(seq_trunc) # Store the number of amino acid residues discarded.
            row['trunc_ratio'] = row['trunc_size'] / len(seq) # Store the truncation size as a ratio. 
            row['original_length'] = len(seq)
            row['seq'] =  seq_trunc
        metadata_trunc_df.append(row)
        
    print(f'truncate_sec: Failed to truncate {n_failures} selenoproteins at the {terminus.upper()}-terminus.')

    ids = pd.Series([i + '-' if (terminus == 'c') else '-' + i for i in metadata_df.index], name='id')
    metadata_trunc_df = pd.DataFrame(metadata_trunc_df, index=ids)
    if drop_failures: # If specified, remove the sequences which could not be truncated.
        metadata_trunc_df = metadata_trunc_df[~metadata_trunc_df.seq.isnull()]

    return metadata_trunc_df


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


class NumpyEncoder(json.JSONEncoder):
    '''Encoder for converting numpy data types into types which are JSON-serializable. Based
    on the tutorial here: https://medium.com/@ayush-thakur02/understanding-custom-encoders-and-decoders-in-pythons-json-module-1490d3d23cf7'''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        # For saving state dicts, which are dictionaries of tensors. 
        if isinstance(obj, OrderedDict):
            new_obj = OrderedDict() # Make a new ordered dictionary. 
            for k, v in obj.items():
                new_obj[k] = v.tolist()
            return new_obj 
        if isinstance(obj, torch.Tensor):
            return obj.tolist()

        return super(NumpyEncoder, self).default(obj)




    









