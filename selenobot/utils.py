'''Utility functions for reading and writing FASTA and CSV files, amongst other things.'''
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
from typing import Dict
import configparser
import pickle
import subprocess


def load_config() -> Dict[str, Dict[str, str]]:
    '''Loads information from the configuration file as a dictionary. Assumes the 
    configuration file is in the project root directory.'''

    # Read in the config file, which is in the project root directory. 
    config = configparser.ConfigParser()
    # with open('/home/prichter/Documents/find-a-bug/find-a-bug.cfg', 'r', encoding='UTF-8') as f:
    with open(os.path.join(os.path.expanduser('~'), 'selenobot.cfg'), 'r', encoding='UTF-8') as f:
        config.read_file(f)
    return config._sections # This gets everything as a dictionary. 


# def load_config_setup():
#     '''Loads everything under the setup flag, making sure to cast the relevant CD-HIT parameters
#     to floats or integers, depending. Assumes all parameters are present.'''
#     setup = load_config()['setup']
#     # Cast the non-string parametes to int and float.
#     setup['cdhit_min_seq_length'] = int(setup['cdhit_min_seq_length'])
#     setup['cdhit_word_length'] = int(setup['cdhit_word_length'])
#     setup['cdhit_sequence_similarity'] = float(setup['cdhit_sequence_similarity'])
#     return setup


def load_config_paths() -> Dict[str, str]:
    '''Loads everything under the [paths] heading in the config file as a dictionary.'''
    return load_config()['paths']


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


def fasta_ids(path, get_id_func=get_id):
    '''Extract all gene IDs stored in a FASTA file.'''
    # Read in the FASTA file as a string. 
    fasta = read(path)
    # Extract all the IDs from the headers, and return the result. 
    ids = [get_id_func(head) for head in re.findall(r'^>.*', fasta, re.MULTILINE)]
    return np.array(ids)


def csv_ids(path):
    '''Extract all gene IDs stored in a CSV file.'''
    df = pd.read_csv(path, usecols=['id']) # Only read in the ID values. 
    return np.ravel(df['id'].values)


def csv_labels(path):
    '''Extract all gene IDs stored in a CSV file.'''
    df = pd.read_csv(path, usecols=['label']) # Only read in the ID values. 
    # Seems kind of bonkers that I need to ravel this. 
    return np.ravel(df.values.astype(np.int32))


def csv_size(path):
    '''Get the number of entries in a FASTA file.'''
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


def pd_from_fasta(path, set_index=False, get_id_func=get_id):
    '''Load a FASTA file in as a pandas DataFrame.'''

    ids = fasta_ids(path, get_id_func=get_id_func)
    seqs = fasta_seqs(path)

    df = pd.DataFrame({'seq':seqs, 'id':ids})
    # df = df.astype({'id':str, 'seq':str})
    if set_index: 
        df = df.set_index('id')
    return df


def pd_to_fasta(df, path=None, textwidth=80):
    '''Convert a pandas DataFrame containing FASTA data to a FASTA file format.'''

    # assert df.index.name == 'id', 'utils.pd_to_fasta: Gene ID must be set as the DataFrame index before writing.'

    fasta = ''
    # for row in tqdm(df.itertuples(), desc='utils.df_to_fasta', total=len(df)):
    for row in df.itertuples():
        fasta += '>|' + str(row.Index) + '|\n'

        # Split the sequence up into shorter, 60-character strings.
        n = len(row.seq)
        seq = [row.seq[i:min(n, i + textwidth)] for i in range(0, n, textwidth)]
        assert len(''.join(seq)) == n, 'utils.pd_to_fasta: Part of the sequence was lmicrobiologyost when splitting into lines.'
        fasta += '\n'.join(seq) + '\n'
    
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




