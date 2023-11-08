'''Setting up the data folder for this project.'''
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import sklearn.cluster
import random
import subprocess
import os
from typing import NoReturn, Dict, List
import sys
import torch
import time

# from transformers import T5EncoderModel, T5Tokenizer
from data_utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_DIR = '/home/prichter/Documents/selenobot/data/'
DETECT_DATA_DIR = '/home/prichter/Documents/selenobot/data/detect/'
GTDB_DATA_DIR = '/home/prichter/Documents/selenobot/data/gtdb/'
UNIPROT_DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot/'

MODEL_NAME = 'Rostlab/prot_t5_xl_half_uniref50-enc'
CD_HIT = '/home/prichter/cd-hit-v4.8.1-2019-0228/cd-hit'

# Parameters for CD-HIT clustering. 
MIN_SEQ_LENGTH = 6 # This is an inclusive lower bound; sequences of length 6 will be present in the dataset 
WORD_LENGTH = 5

# def embed_batch(
#     batch:List[str],
#     model:torch.nn.Module, 
#     tokenizer:T5Tokenizer) -> torch.FloatTensor:
#     '''Embed a single batch, catching any exceptions.
    
#     args:
#         - batch: A list of strings, each string being a tokenized sequence. 
#         - model: The PLM used to generate the embeddings.
#         - tokenizer: The tokenizer used to convert the input sequence into a padded FloatTensor. 
#     '''
#     # Should contain input_ids and attention_mask. Make sure everything's on the GPU. 
#     inputs = {k:torch.tensor(v).to(device) for k, v in tokenizer(batch, padding=True).items()}
#     try:
#         with torch.no_grad():
#             outputs = model(**inputs)
#             return outputs
#     except RuntimeError:
#         print('setup.get_batch_embedding: RuntimeError during embedding for. Try lowering batch size.')
#         return None


# def setup_plm_embeddings(
#     fasta_file_path=None, 
#     embeddings_path:str=None,
#     max_aa_per_batch:int=10000,
#     max_seq_per_batch:int=100,
#     max_seq_length:int=1000) -> NoReturn:
#     '''Generate sequence embeddings of sequences contained in the FASTA file using the PLM specified at the top of the file.
#     Adapted from Josh's code, which he adapted from https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py. 
#     The parameters of this function are designed to prevent GPU memory errors. 
    
#     args:
#         - fasta_file_path: Path to the FASTA file with the input sequences.
#         - out_path: Path to which to write the embeddings.
#         - max_aa_per_batch: The maximum number of amino acid residues in a batch 
#         - max_seq_per_batch: The maximum number of sequences per batch. 
#         - max_seq_length: The maximum length of a single sequence, past which we switch to single-sequence processing
#     '''
#     # Dictionary to store the embedding data. 
#     embeddings = []

#     model = T5EncoderModel.from_pretrained(MODEL_NAME)
#     model = model.to(device) # Move model to GPU
#     model = model.eval() # Set model to evaluation model

#     tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

#     df = pd_from_fasta(fasta_file_path, set_index=False) # Read the sequences into a DataFrame. Don't set the index. 
#     # Need to make sure not to replace all U's with X's in the original DataFrame. 
#     df['seq_standard_aa_only'] = df['seq'].str.replace('U', 'X').replace('Z', 'X').replace('O', 'X') # Replace non-standard amino acids with X token. 

#     # Order the rows according to sequence length to avoid unnecessary padding. 
#     df['length'] = df['seq'].str.len()
#     df = df.sort_values(by='length', ascending=True, ignore_index=True)

#     curr_aa_count = 0
#     curr_batch = []
#     for row in df.itertuples():

#         # Switch to single-sequence processing. 
#         if len(row['seq']) > max_seq_length:
#             outputs = embed_batch([row['seq_standard_aa_only']], model, tokenizer)

#             if outputs is not None:
#                 # Add information to the DataFrame. 
#                 emb = outputs.last_hidden_state[0, :row['length']].mean(dim=0)
#                 embeddings.append(emb)
#             continue

#         curr_batch.append(row['seq_standard_aa_only'])
#         curr_aa_count += row['length']

#         if len(curr_batch) > max_seq_per_batch or curr_aa_count > max_aa_per_batch:
#             outputs = embed_batch(curr_batch, model, tokenizer)

#             if outputs is not None:
#                 for seq, emb in zip(curr_batch, embeddings): # Should iterate over each batch output, or the first dimension. 
#                     emb = emb[:len(seq)].mean(dim=0) # Remove the padding and average over sequence length. 
#                     embeddings.append(emb)

#                     curr_batch = []
#                     curr_aa_count = 0
#     # Remove all unnecessary columns before returning.
#     df = pd.DataFrame(torch.cat(embeddings)).astype(float).drop(columns=['seq_standard_aa_only', 'length'])
#     df.to_csv(embeddings_path)


# data/detect -----------------------------------------------------------------------------------------------------------------------------------------

def setup_train_test_val(
    all_data_path:str=None,
    all_embeddings_path:str=None,
    train_path:str=None,
    test_path:str=None,
    val_path:str=None,
    train_size:int=None,
    test_size:int=None,
    val_size:int=None,
    use_existing_clstr_file:bool=True,
    clstr_file_path:str=os.path.join(UNIPROT_DATA_DIR, 'all_data.clstr')) -> NoReturn:
    '''Splits the data stored in the input path into a train set, test set, and validation set. These sets are disjoint.
    
    args:
        - all_data_path: A path to a FASTA file containing all the UniProt data used for training, testing, etc. the model.
        - all_embeddings_path: A path to a CSV file containing the embeddings of each gene in all_data_path, as well as gene IDs.
        - train_path, test_path, val_path: Paths to the training, test, and validation datasets. 
        - train_size, test_size, val_size: Sizes of the training, test, and validation datasets. 
    '''
    f = 'setup.setup_train_test_val'
    assert (train_size > test_size) and (test_size >= val_size), f'{f}: Expected size order is train_size > test_size >= val_size.'

    # Read in the data, and convert sizes to integers. The index column is gene IDs.
    all_data = pd_from_fasta(all_data_path, set_index=False)

    # This takes too much memory. Will need to do this in chunks. 
    # all_data = all_data.merge(pd.read_csv(all_embeddings_path), on='id')
    # print(f'{f}: Successfully added embeddings to the dataset.') 
    
    # Run CD-HIT on the data stored at the given path, or load an existing clstr_file. .
    clstr_data = pd_from_clstr(clstr_file_path) if use_existing_clstr_file else run_cd_hit(all_data_path, l=MIN_SEQ_LENGTH - 1, n=5) 
    clstr_data_size = len(clstr_data)

    # TODO: Switch over to indices rather than columns, which is faster. 
    # Add the cluster information to the data. 
    all_data = all_data.merge(cluster_data, on='id')
    print(f'{f}: Successfully added homology cluster information to the dataset.') 

    all_data, train_data = sample_homology(all_data, size=len(all_data) - train_size)
    val_data, test_data = sample_homology(all_data, size=val_size)
    
    assert len(train_data) + len(val_data) + len(test_data) == len(clstr_data), f'{f}: Expected {clstr_data_size} sequences present after partitioning, but got {len(train_data) + len(val_data) + len(test_data)}.'

    for data, path in zip([train_data, test_data, val_data], [train_path, test_path, val_path]):
        # Add labels to the DataFrame based on whether or not the gene_id contains a bracket.
        data['label'] = [1 if '[' in gene_id else 0 for gene_id in data.index]
        data.to_csv(path, index=False)
        print(f'{f}: Data successfully written to {path}.')
        add_embeddings_to_file(path, all_embeddings_path)


def add_embeddings_to_file(path:str, embeddings_path:str, chunk_size:int=1000):
    '''Adding embedding information to a CSV file.'''
    f = 'setup.add_embeddings_to_file'

    embeddings_ids = csv_ids(embeddings_path)
    reader = pd.read_csv(path, index_col=['id'], chunksize=chunk_size)
    tmp_file_path = os.path.join(os.path.dirname(path), 'tmp.csv')

    is_first_chunk = True
    n_chunks = csv_size(path) // chunk_size + 1
    for chunk in tqdm(reader, desc=f'{f}: adding embeddings to {path}.', total=n_chunks):
        # Get the indices of the embedding rows corresponding to the data chunk.
        idxs = np.where(np.isin(embeddings_ids, chunk.index, assume_unique=True))[0] + 1 # Make sure to shift the index up by one to include the header. 
        idxs = [0] + list(idxs) # Add the header index for merging. 

        chunk = chunk.merge(pd.read_csv(embeddings_path, skiprows=lambda i : i not in idxs), on='id', how='inner')

        # Subtract 1 from len(idxs) to account for the header row.
        assert len(chunk) == (len(idxs) - 1), f'{f}: Data was lost while merging embedding data.'
        
        chunk.to_csv(tmp_file_path, header=is_first_chunk, mode='w' if is_first_chunk else 'a') # Only write the header for the first file. 
        is_first_chunk = False

    # Replace the old file with tmp. 
    subprocess.run(f'rm {path}', shell=True, check=True)
    subprocess.run(f'mv {tmp_file_path} {path}', shell=True, check=True)


def sample_homology(data:pd.DataFrame, size:int=None):
    '''Subsample the cluster data such that the entirety of any homology group is contained in the sample. 
  
    args:
        - cluster_data: A pandas DataFrame mapping the gene ID to cluster number. 
        - size: The size of the first group, which must also be the smaller group. 
    '''
    f = 'setup.sample_homology'
    assert (len(data) - size) >= size, f'{f}: The size argument must specify the smaller partition. Provided arguments are len(clusters)={len(data)} and size={size}.'

    groups = {'sample':[], 'remainder':[]} # First group is smaller.
    curr_size = 0 # Keep track of how big the sample is, without concatenating DataFrames just yet. 

    ordered_clusters = data.groupby('cluster').size().sort_values(ascending=False).index # Sort the clusters in descending order of size. 

    add_to = 'sample'
    for cluster in tqdm(ordered_clusters, desc=f):
        # If we check for the larger size, it's basically the same as adding to each group in an alternating way, so we need to check for smaller size first.
        cluster = data[data.cluster == cluster]
        if add_to == 'sample' and curr_size < size:
            groups['sample'].append(cluster)
            curr_size += len(cluster)
            add_to = 'remainder'
        else:
            groups['remainder'].append(cluster)
            add_to = 'sample'

    sample, remainder = pd.concat(groups['sample']), pd.concat(groups['remainder'])
    assert len(sample) + len(remainder) == len(data), f"{f}: The combined sizes of the partitions do not add up to the size of the original data."
    assert len(sample) < len(remainder), f'{f}: The sample DataFrame should be smaller than the remainder DataFrame.'

    print(f'{f}: Collected homology-controlled sample of size {len(sample)} ({np.round(len(sample)/len(data), 2) * 100} percent of the input dataset).')
    return sample, remainder


def run_cd_hit(fasta_file_path:str, c:float=0.8, l:int=1, n:int=2) -> pd.DataFrame:
    '''Run CD-HIT on the FASTA file stored in the input path, generating the homology-based sequence similarity clusters.
    
    args:
        - fasta_file_path
        - c: The similarity cutoff for sorting sequences into clusters (see CD-HIT documentation for more information). 
        - l: Minimum sequence length allowed by the clustering algorithm. Note that some truncated sequences
            are filtered out by this parameter (see CD-HIT documentation for more information). 
        - n: Word length (see CD-HIT documentation for more information).
    '''
    # assert (min([len(seq) for seq in fasta_seqs(path)])) > l, 'Minimum sequence length {l + 2} is longer than the shortest sequence.'
    directory, filename = os.path.split(fasta_file_path)
    filename = filename.split('.')[0] # Remove the extension from the filename. 
    subprocess.run(f"{CD_HIT} -i {fasta_file_path} -o {os.path.join(directory, filename)} -c {c} -l {l} -n {n}", shell=True, check=True) # Run CD-HIT.
    subprocess.run(f'rm {os.path.join(directory, filename)}', shell=True, check=True) # Remove the output file with the cluster reps. 

    # Load the clstr data into a DataFrame and return. 
    return pd_from_clstr(os.path.join(directory, filename + '.clstr'))


def setup_detect():
    '''Creates a dataset for the detection classification task, which involves determining
    whether a protein, truncated or not, is a selenoprotein. The data consists of truncated and
    non-truncated selenoproteins, as well as a number of normal full-length proteins equal to all 
    selenoproteins.'''
    
    all_data_size = fasta_size(os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'))

    train_size = int(0.8 * all_data_size)
    test_size = int(0.6 * (all_data_size - train_size)) # Making sure test_size is larger than val_size.
    val_size = all_data_size - train_size - test_size
    sizes = {'train_size':train_size, 'test_size':test_size, 'val_size':val_size}

    # train_data, test_data, and val_data should have sequence information. 
    setup_train_test_val(
        all_data_path=os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'),
        all_embeddings_path=os.path.join(UNIPROT_DATA_DIR, 'all_embeddings.csv'),
        test_path=os.path.join(DETECT_DATA_DIR, 'test.csv'),
        train_path=os.path.join(DETECT_DATA_DIR, 'train.csv'),
        val_path=os.path.join(DETECT_DATA_DIR, 'val.csv'), **sizes)


# /data/uniprot -----------------------------------------------------------------------------------------------------------------------------

def setup_sec_truncated(
    sec_path:str=None, 
    sec_truncated_path:str=None, 
    first_sec_only:bool=True) -> NoReturn:
    '''Truncate the selenoproteins stored in the input file. This function assumes that all sequences contained
    in the file contain selenocysteine, labeled as U.
    
    args:
        - sec_path: A path to the FASTA file containing all of the selenoproteins.
        - sec_truncated_path: The path to write the new FASTA file to, containing the truncated proteins. 
        - first_sec_only: Whether or not to truncate at the first selenocystein residue only. If false, truncation is sequential.
    '''
    # Load the selenoproteins into a pandas DataFrame. 
    df = pd_from_fasta(sec_path, set_index=False)

    df_trunc = {'id':[], 'seq':[]}
    for row in df.itertuples():

        seq = row.seq.split('U')

        if first_sec_only:
            df_trunc['id'].append(row.id + '[1]')
            df_trunc['seq'].append(seq[0])
        else:
            # Number of U's in sequence should be one fewer than the length of the split sequence. 
            df_trunc['id'] += [row.id + f'[{i + 1}]' for i in range(len(seq) - 1)]
            df_trunc['seq'] += ['U'.join(seq[i:i + 1]) for i in range(len(seq) - 1)]
    
    # Do we want the final DataFrame to also contain un-truncated sequences?
    # df = pd.concat([df, pd.DataFrame(df_trunc)]).set_index('id')
    df = pd.DataFrame(df_trunc).set_index('id')
    pd_to_fasta(df, path=sec_truncated_path)


def setup_sprot(sprot_path:str=None) -> NoReturn:
    '''Preprocessing for the SwissProt sequence data. Removes all selenoproteins from the SwissProt database.

    args:
        - sprot_path: Path to the SwissProt FASTA file.
    '''
    f = 'setup.set_sprot'
    sprot_data = pd_from_fasta(sprot_path)
    # Remove all sequences containing U (selenoproteins) from the SwissProt file. 
    selenoproteins = sprot_data['seq'].str.contains('U')
    if np.sum(selenoproteins) == 0:
        print(f'{f}: No selenoproteins found in SwissProt.')
    else:
        print(f'{f}: {np.sum(selenoproteins)} detected out of {len(sprot_data)} total sequences in SwissProt.')
        sprot_data = sprot_data[~selenoproteins]
        assert len(sprot_data) + np.sum(selenoproteins) == fasta_size(sprot_path), f'{f}: Selenoprotein removal unsuccessful.'
        print(f'{f}: Selenoproteins successfully removed from SwissProt.')
        # Overwrite the original file. 
        pd_to_fasta(sprot_data, path=sprot_path)


def setup_uniprot():
    '''Set up all data in the uniprot subdirectory.'''
    
    # Setup the file of truncated selenoproteins. 
    setup_sec_truncated(sec_path=os.path.join(UNIPROT_DATA_DIR, 'sec.fasta'), sec_truncated_path=os.path.join(UNIPROT_DATA_DIR, 'sec_truncated.fasta'))
    setup_sprot(sprot_path=os.path.join(UNIPROT_DATA_DIR, 'sprot.fasta'))
    
    # Combine all FASTA files required for the project into a single all_data.fasta file. 
    fasta_concatenate([os.path.join(UNIPROT_DATA_DIR, 'sec_truncated.fasta'), os.path.join(UNIPROT_DATA_DIR, 'sprot.fasta')], out_path=os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'))
    # setup_plm_embeddings(fasta_file_path=os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta'), embeddings_path=os.path.join(UNIPROT_DATA_DIR, 'all_embeddings.csv'))


# data/gtdb -------------------------------------------------------------------------------------------------------------


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

    # Need to create all embeddings. 

# ------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # setup_uniprot()
    # setup_detect()
    # setup_gtdb(
    pass