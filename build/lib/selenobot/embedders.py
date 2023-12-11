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



