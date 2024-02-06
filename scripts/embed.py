'''A script for embedding protein sequences in a FASTA file. This should be run on the HPC.'''
import sys
# Make the modules in the selenobot directory visible from this script. 
sys.path.append('../selenobot/')

from embedders import PLMEmbedder
from classifiers import Dataset
from utils import dataframe_from_fasta
import subprocess

def add_embeddings(path:str, chunk_size:int=1000) -> NoReturn:
    '''Add embedding information to a dataset, and overwrite the original dataset with the
    modified dataset (with PLM embeddings added).
    
    :param path: The path to the dataset.
    :chunk_size: The size of the chunks to split the dataset into for processing.
    '''
    embedding_ids = pd.read_csv('embeddings.csv', usecols=['id'])['id'].values.ravel() # Read the IDs in the embedding file to avoid loading the entire thing into memory.
    reader = pd.read_csv(path, index_col=['id'], chunksize=chunk_size) # Use read_csv to load the dataset one chunk at a time. 
    tmp_file_path = f'{DATA_DIR}tmp.csv' # The path to the temporary file to which the modified dataset will be written in chunks.

    is_first_chunk = True
    n_chunks = csv_size(path) // chunk_size + 1
    for chunk in tqdm(reader, desc='embed.add_embeddings', total=n_chunks):
        # Get the indices of the embedding rows corresponding to the data chunk. Make sure to shift the index up by one to account for the header. 
        idxs = np.where(np.isin(embedding_ids, chunk.index, assume_unique=True))[0] + 1 
        idxs = [0] + list(idxs) # Add the header index so the column names are included. 
        # Read in the embedding rows, skipping rows which do not match a gene ID in the chunk. 
        chunk = chunk.merge(pd.read_csv('embeddings.csv', skiprows=lambda i : i not in idxs), on='id', how='inner')
        # Check to make sure the merge worked as expected. Subtract 1 from len(idxs) to account for the header row.
        assert len(chunk) == (len(idxs) - 1), f'Data was lost while merging embedding data.'
        
        chunk.to_csv(tmp_file_path, header=is_first_chunk, mode='w' if is_first_chunk else 'a') # Only write the header for the first file. 
        is_first_chunk = False
    # Replace the old dataset with the temporary file. 
    subprocess.run(f'rm {path}', shell=True, check=True)
    subprocess.run(f'mv {tmp_file_path} {path}', shell=True, check=True)


def main(in_path:str, out_path:str):
    '''Generate PLM embeddings for sequences in the specified FASTA file.

    :param in_path: The path to the FASTA file containing the amino acid sequences. 
    :param out_path: The path to which to write the new dataset, with embeddings. 
    '''
    
    # Instantiate the PLM embedder with the model name. 
    embedder = PLMEmbedder('Rostlab/prot_t5_xl_half_uniref50-enc')

    df = dataframe_from_fasta(in_path) # Load the FASTA file containing the sequences to embed. 
    df.to_csv(out_path) # Write the DataFrame to a CSV file. 

    seqs = list(df['seq'].values) # Get the amino acid sequences in the file as a list of strings. 
    ids = list(df['id'].values)

    # Write the embeddings and the corresponding IDs to a CSV output file. This file is temporary. 
    embeddings, ids = embedder(seqs, ids)
    embeddings_df = pd.DataFrame(embeddings)
    embeddings['id'] = ids
    pd.to_csv(embeddings.set_index('id'), 'embeddings.csv')

    # Add the embeddings contained in embeddings.csv to the CSV at out_path.abs
    add_embeddings(out_path)

    # Remove the embeddings.csv file, which is now redundant. 
    subprocess.run('rm embeddings.csv', shell=True, check=True)



