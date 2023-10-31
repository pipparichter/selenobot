'''Setting up the data folder for this project.'''
# import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from datetime import date
import re
import sklearn.cluster
import random
import subprocess
import os
from typing import NoReturn, Dict, List
import sys

from transformers import T5EncoderModel, T5Tokenizer
import torch
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

DATA_DIR = '/home/prichter/Documents/selenobot/data/'
DETECT_DATA_DIR = '/home/prichter/Documents/selenobot/data/detect/'
GTDB_DATA_DIR = '/home/prichter/Documents/selenobot/data/gtdb/'
UNIPROT_DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot/'

MODEL_NAME = 'Rostlab/prot_t5_xl_half_uniref50-enc'
CD_HIT = '/home/prichter/cd-hit-v4.8.1-2019-0228/cd-hit'

MIN_SEQ_LENGTH = 6


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


def embed_batch(
    batch:List[str],
    model:torch.nn.Module, 
    tokenizer:T5Tokenizer) -> torch.FloatTensor:
    '''Embed a single batch, catching any exceptions.
    
    args:
        - batch: A list of strings, each string being a tokenized sequence. 
        - model: The PLM used to generate the embeddings.
        - tokenizer: The tokenizer used to convert the input sequence into a padded FloatTensor. 
    '''
    # Should contain input_ids and attention_mask. Make sure everything's on the GPU. 
    inputs = {k:torch.tensor(v).to(device) for k, v in tokenizer(batch, padding=True).items()}
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs
    except RuntimeError:
        print('setup.get_batch_embedding: RuntimeError during embedding for. Try lowering batch size.')
        return None


def setup_plm_embeddings(
    fasta_file_path=None, 
    embeddings_path:str=None,
    max_aa_per_batch:int=10000,
    max_seq_per_batch:int=100,
    max_seq_length:int=1000) -> NoReturn:
    '''Generate sequence embeddings of sequences contained in the FASTA file using the PLM specified at the top of the file.
    Adapted from Josh's code, which he adapted from https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py. 
    The parameters of this function are designed to prevent GPU memory errors. 
    
    args:
        - fasta_file_path: Path to the FASTA file with the input sequences.
        - out_path: Path to which to write the embeddings.
        - max_aa_per_batch: The maximum number of amino acid residues in a batch 
        - max_seq_per_batch: The maximum number of sequences per batch. 
        - max_seq_length: The maximum length of a single sequence, past which we switch to single-sequence processing
    '''
    # Dictionary to store the embedding data. 
    embeddings = []

    model = T5EncoderModel.from_pretrained(MODEL_NAME)
    model = model.to(device) # Move model to GPU
    model = model.eval() # Set model to evaluation model

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

    df = pd_from_fasta(fasta_file_path, set_index=False) # Read the sequences into a DataFrame. Don't set the index. 
    # Need to make sure not to replace all U's with X's in the original DataFrame. 
    df['seq_standard_aa_only'] = df['seq'].str.replace('U', 'X').replace('Z', 'X').replace('O', 'X') # Replace non-standard amino acids with X token. 

    # Order the rows according to sequence length to avoid unnecessary padding. 
    df['length'] = df['seq'].str.len()
    df = df.sort_values(by='length', ascending=True, ignore_index=True)

    curr_aa_count = 0
    curr_batch = []
    for row in df.itertuples():

        # Switch to single-sequence processing. 
        if len(row['seq']) > max_seq_length:
            outputs = embed_batch([row['seq_standard_aa_only']], model, tokenizer)

            if outputs is not None:
                # Add information to the DataFrame. 
                emb = outputs.last_hidden_state[0, :row['length']].mean(dim=0)
                embeddings.append(emb)
            continue

        curr_batch.append(row['seq_standard_aa_only'])
        curr_aa_count += row['length']

        if len(curr_batch) > max_seq_per_batch or curr_aa_count > max_aa_per_batch:
            outputs = embed_batch(curr_batch, model, tokenizer)

            if outputs is not None:
                for seq, emb in zip(curr_batch, embeddings): # Should iterate over each batch output, or the first dimension. 
                    emb = emb[:len(seq)].mean(dim=0) # Remove the padding and average over sequence length. 
                    embeddings.append(emb)

                    curr_batch = []
                    curr_aa_count = 0
    # Remove all unnecessary columns before returning.
    df = pd.DataFrame(torch.cat(embeddings)).astype(float).drop(columns=['seq_standard_aa_only', 'length'])
    df.to_csv(embeddings_path)


# data/detect -----------------------------------------------------------------------------------------------------------------------------------------


def setup_train_test_val(
    all_data_path:str=None,
    all_embeddings_path:str=None,
    train_path:str=None,
    test_path:str=None,
    val_path:str=None,
    train_size:int=None,
    test_size:int=None,
    val_size:int=None) -> NoReturn:
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
    
    # Run CD-HIT on the data stored at the given path.
    # cluster_data = run_cd_hit(all_data_path, l=MIN_SEQ_LENGTH - 1, n=5)
    cluster_data = pd_from_clstr(os.path.join(UNIPROT_DATA_DIR, 'all_data.clstr'))

    # TODO: Switch over to indices rather than columns, which is faster. 
    # Add the cluster information to the data. 
    all_data = all_data.merge(cluster_data, on='id')
    print(f'{f}: Successfully added homology cluster information to the dataset.') 

    all_data, train_data = sample_homology(all_data, size=len(all_data) - train_size)
    val_data, test_data = sample_homology(all_data, size=val_size)
    
    assert len(np.unique(np.concatenate([train_data.index, test_data.index, val_data.index]))) == len(np.concatenate([train_data.index, test_data.index, val_data.index])), f'{f}: Some proteins are represented more than once in the partitioned data.'

    for data, path in zip([train_data, test_data, val_data], [train_path, test_path, val_path]):
        data = data.set_index('id')
        # Add labels to the DataFrame based on whether or not the gene_id contains a bracket.
        data['label'] = [1 if '[' in gene_id else 0 for gene_id in data.index]
        data.to_csv(path)
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


def pd_from_clstr(clstr_file_path):
    '''Convert a .clstr file string to a pandas DataFrame. The resulting 
    DataFrame maps cluster label to gene ID.'''

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

def setup_summary(log_file_path=None):
    
    if log_file_path is not None: # Write the summary to a log file if specified. 
        with open(log_file_path, 'w', encoding='UTF-8') as f:
            sys.stdout = f

    for file in os.listdir(DETECT_DATA_DIR):
        path = os.path.join(DETECT_DATA_DIR, file)
        print(f'[{file}]')
        print('size:', csv_size(path))
        sec_content = pd.read_csv(path, usecols=['label'])['label'].values.mean()
        print('selenoprotein content:', np.round(sec_content, 3))
        print()

    print('[all_data.fasta]')
    path = os.path.join(UNIPROT_DATA_DIR, 'all_data.fasta')
    seqs = fasta_seqs(path)
    ids = fasta_ids(path)
    print('total sequences:', len(seqs))
    print('total selenoproteins:', len([i for i in ids if '[' in i]))
    print(f'sequences of length >= {MIN_SEQ_LENGTH}:', np.sum(np.array([len(s) for s in seqs]) >= MIN_SEQ_LENGTH))
    print(f'selenoproteins of length >= {MIN_SEQ_LENGTH - 1}:', len([i for i, s in zip(ids, seqs) if '[' in i and len(s) >= MIN_SEQ_LENGTH]))
    print()


if __name__ == '__main__':
    setup_summary()
    # setup_uniprot()
    # setup_detect()
    # setup_gtdb()


# def setup_sprot_sampled(
#     sprot_path:sr=None, 
#     sprot_sampled_path:str=None, 
#     size:int=None) -> NoReturn:
#     '''Filters the SwissProt database by selecting representative proteins. The approach for doing this
#     is based on K-means clustering the embedding data, and sampling from each cluster.

#     args:

#     '''
#     f = 'setup.setup_sprot_sampled'
 
#     sprot_data = pd_from_fasta(path, set_index=False)
#     sprot_data = sprot_data[~sprot_data['seq'].str.contains('U')] # Remove all sequences containing selenocysteine.
#     print(f'{f}: Beginning SwissProt down-sampling. Approximately {size} sequences being sampled without replacement from a pool of {len(sprot_data)}.')
#     sprot_data = sprot_data.sample(size)
#     print(f'{f}: {len(data)} sequences successfully sampled from SwissProt using K-Means clustering.')
#     # Filter the data, and write the filtered data as a FASTA file. 
#     pd_to_fasta(sprot_data.set_index('id'), path=sprot_sampled_path)

# def setup_sprot_sampled(
#     sprot_path=None, 
#     sprot_embeddings_path=None, 
#     sprot_sampled_path=None, 
#     size=500000,
#     n_clusters=500):
#     '''Filters the SwissProt database by selecting representative proteins. The approach for doing this
#     is based on K-means clustering the embedding data, and sampling from each cluster.

#     args:

#     '''
#     f = 'setup.setup_sprot_sampled'
#     # Read in the FASTA file. 
#     data = pd_from_fasta(path, set_index=False)

#     if verbose: print(f'{f}: Beginning SwissProt down-sampling. Approximately {size} sequences being sampled without replacement from a pool of {len(data)}.')
    
#     # Read in the embeddings based on the filepath given as input. 
#     directory, filename = os.path.split(path)
#     embeddings_path = os.path.join(directory, filename.split('.')[0] + '_embeddings.csv')

#     # kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init='auto')
#     kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters, n_init='auto')
#     kmeans.fit(data.values)
#     data['kmeans'] = kmeans.labels_ # Add kmeans labels to the data. 
    
#     # assert np.all(data.groupby('kmeans').size() >= size//n_clusters), f'setup.setup_sprot_sampled: Trying to sample too many elements from a kmeans cluster.'
#     data = data.groupby('kmeans').sample(size//n_clusters, replace=True)
#     data = data.drop_duplicates(subset=['id'], ignore_index=True)

#     print(f'{f}: {len(data)} sequences successfully sampled from SwissProt using K-Means clustering.')

#     # Remove all sequences containing selenocysteine. 
#     data = data[~data['seq'].str.contains('U')]

#     # Filter the data, and write the filtered data as a FASTA file. 
#     pd_to_fasta(data.set_index('id'), path=sprot_sampled_path)


# def h5_to_csv(path, batch_size=100):
#     '''Convert a data file in HD5 format to a CSV file.'''

#     file_ = h5py.File(path)
#     keys = list(file_.keys())

#     batches = [keys[i:min(i + batch_size, len(keys))] for i in range(0, len(keys), batch_size)]

#     header = True
#     for batch in tqdm(batches, desc='h5_to_csv'):
#         data = np.array([file_[k][()] for k in batch])
#         df = pd.DataFrame(data)
#         df['id'] = [get_id(k) for k in batch]

#         # df = df.set_index('id')
#         df.to_csv(path.split('.')[0] + '.csv', mode='a', header=header)

#         # After the first loop, don't write the headers to the file. 
#         header = False

#     file_.close()

