'''This file contains a series of classes which allow for the creation of embedding datasets
from sequence data.'''

# import transformers
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from typing import List

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class AacEmbedder():

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


class PlmEmbedder():
    '''Adapted from Josh's code, which he adapted from https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py'''

    def __init__(model_name:str):
        '''Initializes a PLM embedder object.
        
        :param model_name: The name of the pre-trained PLM to load from HuggingFace.
        '''
        
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model = model.to(DEVICE) # Move model to GPU
        self.model = model.eval() # Set model to evaluation model
        # Should be a T5Tokenizer object. 
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)


    def __call__(self, data:List[SyntaxError], path:str,
        max_aa_per_batch:int=10000,
        max_seq_per_batch:int=100,
        max_seq_length:int=1000) -> torch.FloatTensor:
        '''
        Embeds the input data using the PLM stored in the model attribute.
        
        :param data: 
        :param path: Path to which to write the embeddings.
        :param max_aa_per_batch: The maximum number of amino acid residues in a batch 
        :param max_seq_per_batch: The maximum number of sequences per batch. 
        :param max_seq_length: The maximum length of a single sequence, past which we switch to single-sequence processing
        :return: A Tensor object containing all of the embedding data. 
        '''

        embeddings = [] # List to store the embedding data. 
        data = [seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X') for seq in data] # Replace non-standard amino acids with X token.
        # Order the sequences in ascending order according to sequence length to avoid unnecessary padding. 
        data = sorted(data, key=len)

        curr_aa_count = 0
        curr_batch = []
        for seq in data:
            # Switch to single-sequence processing if length limit is exceeded.
            if len(seq) > max_seq_length:
                outputs = self.embed_batch([seq])

                if outputs is not None:
                    # Add information to the list. 
                    emb = outputs.last_hidden_state[0, :row['length']].mean(dim=0)
                    embeddings.append(emb)
                continue

            # Add the sequence to the batch, and keep track of total amino acids in the batch. 
            curr_batch.append(seq)
            curr_aa_count += len(seq)

            if len(curr_batch) > max_seq_per_batch or curr_aa_count > max_aa_per_batch:
                # If any of the presepecified limits are exceeded, go ahead and embed the batch. 
                outputs = self.embed_batch(curr_batch)

                if outputs is not None:
                    for seq, emb in zip(curr_batch, outputs): # Should iterate over each batch output, or the first dimension. 
                        emb = emb[:len(seq)].mean(dim=0) # Remove the padding and average over sequence length. 
                        embeddings.append(emb)

                        # Reset the current batch and amino acid count. 
                        curr_batch = []
                        curr_aa_count = 0

        # Concatenate the list of tensors and return. 
        return torch.cat(embeddings).astype(float)


    def embed_batch(self, batch:List[str]) -> torch.FloatTensor:
        '''Embed a single batch, catching any exceptions.
        
        :param batch: A list of strings, each string being a tokenized sequence. 
        :return: A PyTorch tensor containing PLM embeddings for the batch. 
        '''
        # Should contain input_ids and attention_mask. Make sure everything's on the GPU. 
        inputs = {k:torch.tensor(v).to(device) for k, v in tokenizer(batch, padding=True).items()}
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs
        except RuntimeError:
            print('embedders.PlmEmbedder.embed_batch: RuntimeError during embedding. Try lowering batch size.')
            return None



