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


def gtdb_load_genome_metadata(path:str, reps_only:bool=True):
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


def default_output_path(path:str, op:str=None, ext:str=None):
    '''Construct a default output path for a program.'''
    file_name = os.path.basename(path)
    if ext is not None:
        file_name, _ = os.path.splitext(file_name)
        file_name += f'.{op}.{ext}'
    else:
        file_name = f'{file_name}.{op}'
    return os.path.join(os.path.dirname(path), file_name)


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



def digitize(values:np.ndarray, bin_edges:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Assign a bin label to each input value using the specified bin edges.
    
    :param values: An array of values for which to assign bin labels. 
    :param bin_edges: The bin edges. The left and right-most edges are inclusive, and bins[i-1] <= x < bins[i]. 
    :return: A tuple containing (1) an array of size len(values) containing bin labels for each value, and (2) an array
        of size len(bin_edges) - 1 containing names for each bin. 
    '''
    # Numpy digitize does not include the right-most bin edge, but the histogram function does. This
    # leads to an annoying problem where the largest value is assigned an out-of-bounds bin value, unless
    # the right-most bin edge is incremented. 
    bin_edges[-1] = bin_edges[-1] + 1
    bin_labels = np.digitize(values, bin_edges)
    bin_names = [f'{int(bin_edges[i])}-{int(bin_edges[i + 1])}' for i in range(len(bin_edges) - 1)]
    return (bin_labels, bin_names)


def groupby(values:np.ndarray, keys:np.ndarray) -> Dict[int, np.ndarray]:
    '''Group the input array of values according to the corresponding keys.
    
    :param values: An array of values to group. 
    :param: An array of keys with the same shape as the values array. 
    :return: A dictionary mapping each key to a set of corresponding values.     
    '''
    sort_idxs = np.argsort(keys) # Get the indices which would put the keys in order. 
    sorted_values, sorted_keys = values[sort_idxs], keys[sort_idxs] # Sort the values and keys. 
    
    unique_keys, unique_idxs = np.unique(sorted_keys, return_index=True) 
    # unique_idxs will basically contain the index of the first location of each key (i.e. bin)
    binned_values = np.split(sorted_values, unique_idxs)[1:]
    return {int(key):value for key, value in zip(unique_keys, binned_values)}


def sample(values:np.ndarray, hist:np.ndarray, bin_edges:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Sample from a list of values according to the bins and bin heights from a histogram.
    
    :param values: An array of values to sample. 
    :param hist: The number of entries in each histogram bin; expect output from np.histogram. 
    :param bin_edges: The bin edges. The left and right-most edges are inclusive, and bins[i-1] <= x < bins[i]. 
    :return: A tuple containing a subsample of the values, which should follow the distribution of the input histogram,
        as well as the indices of the sample. 
    '''
    bin_labels, _ = digitize(values, bin_edges)
    bin_idxs = groupby(np.arange(len(values)), bin_labels)
    
    # Remove bins which are outside the histogram boundaries (i.e. are less than the minimum bin edge
    # or greater than the maximim bin edge).
    if 0 in bin_idxs:
        del bin_idxs[0]
    if len(bin_edges) in bin_idxs:
        del bin_idxs[len(bin_edges)]

    # Want to take the biggest sample possible while remaining in line with the input hist. 
    scale = min([len(idxs) / hist[label - 1] for label, idxs in bin_idxs.items()])

    sample_idxs = []
    for label, idxs in bin_idxs.items():
        n = int(scale * hist[label - 1])
        sample_idxs.append(np.random.choice(idxs, n, replace=False))
    sample_idxs = np.concat(sample_idxs).ravel()
    
    return (values[sample_idxs], sample_idxs)


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




    









