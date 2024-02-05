'''A script for embedding protein sequences in a FASTA file. This should be run on the HPC.'''
import sys
# Make the modules in the selenobot directory visible from this script. 
sys.path.append('../selenobot/')

from embedders import PLMEmbedder
from utils import dataframe_from_fasta



def main(path:str):
    '''Generate PLM embeddings for sequences in the specified FASTA file.

    :param path: The path to the FASTA file containing the amino acid sequences. 
    '''
    
    # Instantiate the PLM embedder with the model name. 
    embedder = PLMEmbedder('Rostlab/prot_t5_xl_half_uniref50-enc')

    # Load the FASTA file containing the sequences to embed. 
    df = dataframe_from_fasta(path)
    seqs = list(df['seq'].values) # Get the amino acid sequences in the file as a list of strings. 

    embedder(seqs)

