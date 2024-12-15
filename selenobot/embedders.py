'''This file contains a series of classes which allow for the creation of embedding datasets
from sequence data.'''

# import transformers
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
import itertools

from typing import List, Tuple

class LengthEmbedder():
    name = 'len'
    def __init__(self):
        '''Initializes a LengthEmbedder object.'''
        # There is no tokenizer in this case, so leave it as the default None. 
        super(LengthEmbedder, self).__init__()
        self.type = 'len'

    def __call__(self, seqs:List[str], ids:List[str]):
        '''Takes a list of amino acid sequences, and produces a PyTorch tensor containing the lengths
        of each sequence.'''
        lengths = [[len(seq)] for seq in seqs]
        # I think the datatype should be float32, but not totally sure... 
        return np.array(lengths), np.array(ids)


class KmerEmbedder():
    name = 'kmer'
    amino_acids = list('ARNDCQEGHILKMFPOSTWYV')
    # aa_to_int_map = {aa: i for i, aa in enumerate(list('ARNDCQEGHILKMFPOSTWYV'))}

    def __init__(self, k:int=1):
        '''Initializes an AACEmbedder object.'''
        super(AACEmbedder, self).__init__()
        self.type = f'{k}mer'
        self.k = k
        # Sort list of k-mers to ensure consistent ordering
        self.kmers = sorted([''.join(kmer) for kmer in itertools.permutations(KmerEmbedder.amino_acids)])
        self.kmer_to_int_map = {kmer:i for i, kmer in enumerate(self.kmers)}

    def _get_kmers(self, seq:str):
        '''Encode a sequence using k-mers.'''
        assert len(seq) > self.k, f'KmerEmbedder._get_kmers: Input sequence has length {len(seq)}, which is too short for k-mers of size {self.k}.'
        
        kmers = {kmer:0 for kmer in self.kmers}
        for i in range(len(seq) - self.k):
            kmer = seq[i:i + self.k]
            kmers[kmer] += 1
        # Normalize the k-mer counts by sequence length. 
        kmers = {kmer:count / len(seq) for kmer, count in kmers.items()}
        return kmers

    def __call__(self, seqs:List[str], ids:List[str]) -> torch.Tensor:
        '''Takes a list of amino acid sequences, and produces a PyTorch tensor containing the lengths
        of each sequence.'''
        embs = []
        for seq in tqdm(seqs, desc='KmerEmbedder.__call__'):
            embs.append(self._get_kmers(seq))
        embs = pd.DataFrame(embs)
        embs = embs[self.kmers] # Make sure column ordering is consistent. 

        return embs.values, np.array(ids)


class PLMEmbedder():
    '''Adapted from Josh's code, which he adapted from https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py'''
    name = 'plm'

    def __init__(self, model_name:str='Rostlab/prot_t5_xl_half_uniref50-enc', mean_pool:bool=True):
        '''Initializes a PLM embedder object.'''

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean_pool = mean_pool
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.to(self.device) # Move model to GPU.
        self.model.eval() # Set model to evaluation model.
        # Should be a T5Tokenizer object. 
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True) #, cleanup_tokenization_spaces=False)


    def __call__(self, seqs:List[str], ids:List[str], max_aa_per_batch:int=10000, max_seq_per_batch:int=100, max_seq_length:int=1000):
        '''
        Embeds the input data using the PLM stored in the model attribute. Note that this embedding
        algorithm does not preserve the order of the input sequences, so IDs must be included for each sequence.
        
        :param seqs: A list of amino acid sequences to embed.
        :param ids: A list of identifiers for the amino acid sequences. 
        :param max_aa_per_batch: The maximum number of amino acid residues in a batch 
        :param max_seq_per_batch: The maximum number of sequences per batch. 
        :param max_seq_length: The maximum length of a single sequence, past which we switch to single-sequence processing
        :return: A Tensor object containing all of the embedding data. 
        '''
        seqs = [s.replace('U', 'X').replace('Z', 'X').replace('O', 'X') for s in seqs] # Replace non-standard amino acids with X token.
        seqs = list(zip(ids, seqs)) # Store the IDs with the sequences as tuples in a list. 
        # Order the sequences in ascending order according to sequence length to avoid unnecessary padding. 
        seqs = sorted(seqs, key=lambda t : len(t[1]))

        embs = []
        curr_aa_count = 0
        curr_batch = []

        def add(outputs, batch:List[Tuple[int, str]]=None):
            '''Extract the embeddings from model output and mean-pool across the length
            of the sequence. Add the embeddings to the embeddings list.'''
            if outputs is not None:
                for (i, s), e in zip(batch, outputs.last_hidden_state): # Should iterate over each batch output, or the first dimension. 
                    e = e[:len(s)] # Remove the padding. 
                    if self.mean_pool:
                        e = e.mean(dim=0) # If mean pooling is specified, average over sequence length. 
                    embs.append((i, e)) # Append the ID and embedding to the list. 

        for i, s in tqdm(seqs, desc='PLMEmbedder.__call__'):
            # Switch to single-sequence processing if length limit is exceeded.
            if len(s) > max_seq_length:
                outputs = self.embed_batch([s])
                add(outputs, batch=[(i, s)])
                continue

            # Add the sequence to the batch, and keep track of total amino acids in the batch. 
            curr_batch.append((i, s))
            curr_aa_count += len(s)

            if len(curr_batch) > max_seq_per_batch or curr_aa_count > max_aa_per_batch:
                # If any of the presepecified limits are exceeded, go ahead and embed the batch. 
                # Make sure to only pass in the sequence. 
                outputs = self.embed_batch([s for _, s in curr_batch])
                add(outputs, batch=curr_batch)

                # Reset the current batch and amino acid count. 
                curr_batch = []
                curr_aa_count = 0

        # Handles the case in which the minimum batch size is not reached.
        if len(curr_batch) > 0:
            outputs = self.embed_batch([s for _, s in curr_batch])
            add(outputs, batch=curr_batch)

        # Separate the IDs and embeddings in the list of tuples. 
        ids = [i for i, _ in embs]
        embs = [torch.unsqueeze(e, 0) for _, e in embs]
        embs = torch.cat(embs).float()
        return embs.cpu().numpy(), np.array(ids)

    def embed_batch(self, batch:List[str]) -> torch.FloatTensor:
        '''Embed a single batch, catching any exceptions.
        
        :param batch: A list of strings, each string being a tokenized sequence. 
        :return: A PyTorch tensor containing PLM embeddings for the batch. 
        '''
        # Characters in the sequence need to be space-separated, apparently. 
        batch = [' '.join(list(s)) for s in batch]
        # Should contain input_ids and attention_mask. Make sure everything's on the GPU. 
        # The tokenizer defaults mean that add_special_tokens=True and padding=True is equivalent to padding='longest'
        inputs = {k:torch.tensor(v).to(self.device) for k, v in self.tokenizer(batch, padding=True).items()} # , cleanup_tokenization_spaces=True).items()}
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs
        except RuntimeError:
            print('PLMEmbedder.embed_batch: RuntimeError during embedding. Try lowering batch size.')
            return None


def embed(df:pd.DataFrame, path:str=None, append:bool=False, k_values:List[int]=[1, 2, 3, 4]):
    '''Embed the sequences in the input DataFrame (using all three embedding methods), and store the embeddings and metadata in an HDF5
    file at the specified path.'''

    def add(store:pd.HDFStore, key:str, df:pd.DataFrame):
        '''Add a DataFrame to the specified node in the HDF file. If append is specified, append the 
        DataFrame rather than creating a new node.'''
        # NOTE: The table format performs worse, but will enable modification later on. 
        if append:
            # Does not check if data being appended overlaps with existing data in the table, so be careful. 
            store.append(key, df, 'table')
        else:
            store.put(key, df, format='table')


    df = df.sort_index() # Sort the index of the DataFrame to ensure consistent ordering. 
    store = pd.HDFStore(path, mode='a' if append else 'w') # Should confirm that the file already exists. 
    add(store, 'metadata', df)

    embedders = [PLMEmbedder(), LengthEmbedder()]
    embedders += [KmerEmbedder(k=k) for k in k_values]

    for embedder in embedders:
        embs, ids = embedder(df.seq.values.tolist(), df.index.values.tolist())
        sort_idxs = np.argsort(ids)
        embs, ids = embs[sort_idxs, :], ids[sort_idxs]
        # I don't think failing to detach the tensors here is a problem, because it is being converted to a pandas DataFrame. 
        emb_df = pd.DataFrame(embs, index=ids)
        add(store, embedder.type, emb_df) # Make sure it's in table format if I need to append to it later.
        print(f'embed: Embeddings of type {embedder.type} added to HDF file.')

    print(f'embed: Embedding data written to {path}')
    store.close()



