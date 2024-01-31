'''This file contains a series of classes which allow for the creation of embedding datasets
from sequence data.'''

# import transformers
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from typing import List


def check_input_sequences():
    # TODO: A function for checking input sequences. 
    pass

class Embedder():
    '''Base Embedder class, specifies the necessary functions, which is really
    just __init__ and __call__'''
    def __init__(self):
        pass

    def __call__(self, data:List[str]) -> torch.Tensor:
        pass


class PLMEmbedder():
    pass


class LengthEmbedder():

    def __init__(self):
        '''Initializes a LengthEmbedder object.'''
        # There is no tokenizer in this case, so leave it as the default None. 
        super(LengthEmbedder, self).__init__()
        self.type = 'len'

    def __call__(self, data):
        '''Takes a list of amino acid sequences, and produces a PyTorch tensor containing the lengths
        of each sequence.'''
        lengths = [[len(seq)] for seq in data]
        # I think the datatype should be float32, but not totally sure... 
        return torch.Tensor(lengths).to(torch.float32)


class AacEmbedder(Embedder):

    def __init__(self):
        '''Initializes an AacEmbedder object.'''
        super(AacEmbedder, self).__init__()
        self.type = 'aac'

    def __call__(self, data:List[str]) -> torch.Tensor:
        '''Takes a list of amino acid sequences, and produces a PyTorch tensor containing the lengths
        of each sequence.'''
        # aas = 'ARNDCQEGHILKMFPOSUTWYVBZXJ'
        aas = 'ARNDCQEGHILKMFPOSTWYV'
        aa_to_int_map = {aas[i]: i for i in range(len(aas))}

        embs = []

        for seq in data:
            # NOTE: I am pretty much just ignoring all the non-standard amino acids. 
            seq = [aa for aa in seq if aa in aa_to_int_map]
            # assert np.all([aa in aa_to_int_map for aa in seq]), 'embedder.AacEmbedder.__call__: Some amino acids in the input sequences are not present in the amino-acid-to-integer map.'

            # Map each amino acid to an integer using the internal map. 
            # seq = np.array([self.aa_to_int_map[aa] for aa in seq])
            seq = np.array([aa_to_int_map[aa] for aa in seq])
            
            emb = np.zeros(shape=(len(seq), len(aa_to_int_map)))
            emb[np.arange(len(seq)), seq] = 1
            emb = np.sum(emb, axis=0)
            # Now need to normalize according to sequence length. 
            emb = emb / len(seq)
            embs.append(list(emb))

        # encoded_seqs is now a 2D list. Convert to a tensor so it works as a model input. 
        return torch.Tensor(embs).to(torch.float32)

    def __str__(self):
        return 'aac'


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
