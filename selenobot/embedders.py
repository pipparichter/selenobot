'''This file contains a series of classes which allow for the creation of embedding datasets
from sequence data.'''

# import transformers
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
from transformers import T5Tokenizer, T5EncoderModel
import itertools
import re   
from typing import List, Tuple

# NOTE: tqdm progress bars print to standard error for reasons which are not relevant to me... 
# https://stackoverflow.com/questions/75580592/why-is-tqdm-output-directed-to-sys-stderr-and-not-to-sys-stdout 
# I think I want to print to stdout. 

# NOTE: The model I am using is apparently usable on 8GB of RAM, so I am not sure why it is failing. 

class LengthEmbedder():
    name = 'len'
    def __init__(self):
        '''Initializes a LengthEmbedder object.'''
        # There is no tokenizer in this case, so leave it as the default None. 
        # super(LengthEmbedder, self).__init__()
        self.type = 'len'

    def __call__(self, seqs:List[str], ids:List[str]):
        '''Takes a list of amino acid sequences, and produces a PyTorch tensor containing the lengths
        of each sequence.'''
        lengths = [[len(seq)] for seq in seqs]
        # I think the datatype should be float32, but not totally sure... 
        return np.array(lengths), np.array(ids)


class KmerEmbedder():
    name = 'kmer'
    amino_acids = list('ARNDCQEGHILKMFPSTWYV') # Don't include non-standard amino acids. 
    # aa_to_int_map = {aa: i for i, aa in enumerate(list('ARNDCQEGHILKMFPOSTWYV'))}

    def __init__(self, k:int=1):
        '''Initializes an KmerEmbedder object.'''
        # super(KmerEmbedder, self).__init__()
        self.type = f'aa_{k}mer'
        self.k = k
        # Sort list of k-mers to ensure consistent ordering
        self.kmers = sorted([''.join(kmer) for kmer in itertools.permutations(KmerEmbedder.amino_acids, k)])
        self.kmer_to_int_map = {kmer:i for i, kmer in enumerate(self.kmers)}

    def _get_kmers(self, seq:str):
        '''Encode a sequence using k-mers.'''
        # assert len(seq) > self.k, f'KmerEmbedder._get_kmers: Input sequence has length {len(seq)}, which is too short for k-mers of size {self.k}.'
        if len(seq) < self.k:
            print(f'KmerEmbedder._get_kmers: Input sequence has length {len(seq)}, which is too short for k-mers of size {self.k}.', flush=True)
            return None

        kmers = {kmer:0 for kmer in self.kmers}
        for i in range(len(seq) - self.k):
            kmer = seq[i:i + self.k]
            if kmer in kmers:
                kmers[kmer] += 1

        # Normalize the k-mer counts by sequence length. 
        kmers = {kmer:count / len(seq) for kmer, count in kmers.items()}
        return kmers

    def __call__(self, seqs:List[str], ids:List[str]) -> torch.Tensor:
        '''Takes a list of amino acid sequences, and produces a PyTorch tensor containing the lengths
        of each sequence.'''
        seqs = [s.replace('U', 'X').replace('Z', 'X').replace('O', 'X') for s in seqs] # Replace non-standard amino acids with X token.
        embs = []

        for id_, seq in tqdm(list(zip(ids, seqs)), desc='KmerEmbedder.__call__', file=sys.stdout):
            emb = self._get_kmers(seq)
            if emb is not None:
                embs.append((id_, emb))

        # Convert list of tuples to a DataFrame for processing. 
        embs = pd.DataFrame([e for i, e in embs], index=[i for i, e in embs])
        embs = embs[self.kmers] # Make sure column ordering is consistent. 

        return embs.values, embs.index.values


class PLMEmbedder():
    '''Adapted from Josh's code, which he adapted from https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py'''
    name = 'plm'

    def __init__(self, model_name:str='Rostlab/prot_t5_xl_half_uniref50-enc', mean_pool:bool=True, half_precision:bool=True):
        '''Initializes a PLM embedder object.'''
        self.type = 'plm'
        self.dtype = torch.float16 if half_precision else torch.float32
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean_pool = mean_pool
        print(f'PLMEmbedder.__init__: Loading pre-trained model {model_name}.', flush=True)
        self.model = T5EncoderModel.from_pretrained(model_name, torch_dtype=self.dtype)
        # print(f'PLMEmbedder.__init__: Model loaded successfully.', flush=True)
        self.model.to(self.device) # Move model to GPU.
        self.model.eval() # Set model to evaluation model.
        # Should be a T5Tokenizer object. 
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True, clean_up_tokenization_spaces=True) 


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

        for i, s in tqdm(seqs, desc='PLMEmbedder.__call__', file=sys.stdout):
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
        inputs = {k:torch.tensor(v).to(self.device) for k, v in self.tokenizer(batch, padding=True).items()} 
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs
        except RuntimeError:
            print('PLMEmbedder.embed_batch: RuntimeError during embedding. Try lowering batch size.', flush=True)
            return None


def embed(df:pd.DataFrame, path:str=None, overwrite:bool=False, embedders:List=[], format_='table'): 
    '''Embed the sequences in the input DataFrame (using all three embedding methods), and store the embeddings and metadata in an HDF5
    file at the specified path.'''

    store = pd.HDFStore(path, mode='a' if (not overwrite) else 'w') # Should confirm that the file already exists. 
    existing_keys = [key.replace('/', '') for key in store.keys()]

    df = df.sort_index() # Sort the index of the DataFrame to ensure consistent ordering. 
    seq_is_nan = df.seq.isnull()
    print(f'embed: Removing {np.sum(seq_is_nan)} null entries from the sequence DataFrame. {len(df) - np.sum(seq_is_nan)} sequences remaining.', flush=True)
    df = df[~seq_is_nan]

    if 'metadata' not in existing_keys:
        # Avoid mixed column data types. 
        string_cols = [col for col in df.columns if (df[col].dtype == 'object')] 
        df[string_cols] = df[string_cols].fillna('None')
        store.put('metadata', df, format=format_, data_columns=None)

    for embedder in embedders:
        if (embedder.type in existing_keys) and (not overwrite):
            print(f'Embeddings of type {embedder.type} are already present in {path}')
        else:
            embs, ids = embedder(df.seq.values.tolist(), df.index.values.tolist())
            sort_idxs = np.argsort(ids)
            embs, ids = embs[sort_idxs, :], ids[sort_idxs]
            # I don't think failing to detach the tensors here is a problem, because it is being converted to a pandas DataFrame. 
            emb_df = pd.DataFrame(embs, index=ids)
            store.put(embedder.type, emb_df, format=format_, data_columns=None) 
            print(f'embed: Embeddings of type {embedder.type} added to {path}.', flush=True)

    store.close()



