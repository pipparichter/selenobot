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
from transformers import EsmTokenizer, EsmModel, AutoTokenizer, EsmForMaskedLM
import itertools
import re   
from typing import List, Tuple

# Might need to make the batch allocations vary less. Maybe divide sequences into equal-sized blocks. 
# https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
# https://discuss.pytorch.org/t/unable-to-allocate-cuda-memory-when-there-is-enough-of-cached-memory/33296 
# https://alexdremov.me/simple-ways-to-speedup-your-pytorch-model-training/ 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'expandable_segments:True'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:50'

# TODO: Check on tokenizer behavior... are sequences ever truncated?

# NOTE: tqdm progress bars print to standard error for reasons which are not relevant to me... 
# https://stackoverflow.com/questions/75580592/why-is-tqdm-output-directed-to-sys-stderr-and-not-to-sys-stdout 
# I think I want to print to stdout. Instead of using the pooler_ouput layer, I could try manually extracting
# the CLS token from the last hidden state. 
# Maybe it is also related to this error? 
    # Some weights of EsmModel were not initialized from the model checkpoint at facebook/esm2_t36_3B_UR50D and are newly initialized: 
    # ['esm.pooler.dense.bias', 'esm.pooler.dense.weight']
    # You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

# NOTE: The model I am using is apparently usable on 8GB of RAM, so I am not sure why it is failing. 

class LengthEmbedder():
    def __init__(self):
        '''Initializes a LengthEmbedder object.'''
        # There is no tokenizer in this case, so leave it as the default None. 
        # super(LengthEmbedder, self).__init__()
        pass 

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

    def __init__(self, model=None, tokenizer=None, checkpoint:str=None):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.from_pretrained(checkpoint)
        self.model.to(self.device) # Move model to GPU.
        self.model.eval() # Set model to evaluation model.

        self.tokenizer = tokenizer.from_pretrained(checkpoint, do_lower_case=False, legacy=True, clean_up_tokenization_spaces=True)


    def embed_batch(self, seqs:List[str]) -> torch.FloatTensor:
        '''Embed a single batch, catching any exceptions.
        
        :param batch: A list of strings, each string being a tokenized sequence. 
        :return: A PyTorch tensor containing PLM embeddings for the batch. 
        '''
        # Should contain input_ids and attention_mask. Make sure everything's on the GPU. 
        # The tokenizer defaults mean that add_special_tokens=True and padding=True is equivalent to padding='longest'
        inputs = {k:torch.tensor(v).to(self.device) for k, v in self.tokenizer(seqs, padding=True).items()}
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs
        except RuntimeError:
            return None

    @staticmethod
    def sort(seqs, ids):
        ''''''
        seqs_and_ids = list(zip(seqs, ids)) # Store the IDs with the sequences as tuples in a list. 
        seqs_and_ids = sorted(seqs_and_ids, key=lambda x : len(x[1]))[::-1]
        return seqs_and_ids
        
    def __call__(self, seqs:List[str], ids:List[str], max_aa_per_batch:int=1000):
        '''
        Embeds the input data using the PLM stored in the model attribute. Note that this embedding
        algorithm does not preserve the order of the input sequences, so IDs must be included for each sequence.
        
        :param seqs: A list of amino acid sequences to embed.
        :param ids: A list of identifiers for the amino acid sequences. 
        :param max_aa_per_batch: The maximum number of amino acid residues in a batch  
        '''
        seqs = self._preprocess(seqs)
        pbar = tqdm(PLMEmbedder.sort(seqs, ids), desc='PLMEmbedder.__call__', file=sys.stdout)

        embs, ids = [], [] 

        batch_seqs, batch_ids, aa_count, errors = [], [], 0, 0
        for s, i in pbar:
            pbar.set_description(f'PLMEmbedder.__call__: {errors} sequences skipped.')

            batch_seqs.append(s)
            batch_ids.append(i)
            aa_count += len(s)

            if aa_count > max_aa_per_batch:
                outputs = self.embed_batch(batch_seqs)
                if outputs is not None:
                    embs += self._postprocess(outputs, seqs=batch_seqs)
                    ids += batch_ids
                else:
                    errors += 1
                batch_ids, batch_seqs, aa_count = [], [], 0

        # Handles the case in which the minimum batch size is not reached.
        if aa_count > 0:
                outputs = self.embed_batch(batch_seqs)
                if outputs is not None:
                    embs += self._postprocess(outputs, seqs=batch_seqs)
                    ids += batch_ids

        pbar.close()
        embs = torch.cat([torch.unsqueeze(emb, 0) for emb in embs]).float()
        return embs.numpy(), np.array(ids)



class ProtT5Embedder(PLMEmbedder):

    checkpoint = 'Rostlab/prot_t5_xl_half_uniref50-enc'

    def __init__(self):

        super(ProtT5Embedder, self).__init__(model=T5EncoderModel, tokenizer=T5Tokenizer, checkpoint=ProtT5Embedder.checkpoint)

    def _preprocess(self, seqs:List[str]) -> List[str]:
        ''''''
        seqs = [seq.replace('*', '') for seq in seqs]
        seqs = [seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', '') for seq in seqs] # Replace rare amino acids with X token.
        seqs = [' '.join(list(seq)) for seq in seqs] # Characters in the sequence need to be space-separated, apparently. 
        return seqs  

    def _postprocess(self, outputs, seqs:List[str]=None) -> List[torch.FloatTensor]:
        ''''''
        seqs = [''.join(seq.split()) for seq in seqs] # Remove the added whitespace so length is correct. 

        outputs = outputs.last_hidden_state.cpu()
        outputs = [emb[:len(seq)] for emb, seq in zip(outputs, seqs)]
        outputs = [emb.mean(dim=0) for emb in outputs] # Take the average over the sequence length. 
        return outputs 


class ESMEmbedder(PLMEmbedder):

    @staticmethod
    def _pooler_gap(emb:torch.FloatTensor, seq:str) -> torch.FloatTensor:
        emb = emb[1:len(seq) + 1]
        emb = emb.mean(dim=0)
        return emb 

    @staticmethod
    def _pooler_cls(emb:torch.FloatTensor, *args) -> torch.FloatTensor:
        return emb[0] # Extract the CLS token, which is the first element of the sequence. 

    # # checkpoint = 'facebook/esm2_t36_3B_UR50D'
    # # checkpoint = 'facebook/esm2_t33_650M_UR50D'

    def __init__(self, method:str='gap'):
        # checkpoint = 'facebook/esm2_t33_650M_UR50D'
        checkpoint = 'facebook/esm2_t36_3B_UR50D'
        models = {'gap':EsmModel, 'log':EsmForMaskedLM, 'cls':EsmModel}
        poolers = {'gap':ESMEmbedder._pooler_gap, 'cls':ESMEmbedder._pooler_cls} 

        super(ESMEmbedder, self).__init__(model=models[method], tokenizer=AutoTokenizer, checkpoint=checkpoint)
        self.method = method 
        self.pooler = poolers.get(method, None)

    def _preprocess(self, seqs:List[str]):
        '''Based on the example Jupyter notebook, it seems as though sequences require no real pre-processing for the ESM model.'''
        seqs = [seq.replace('*', '') for seq in seqs]
        # Don't need to do this, there is already an end-of-sequence token that has a value of 1 in the attention mask. 
        # if self.method == 'log':
        #     seqs = [seq + ESMEmbedder.unknown_token for seq in seqs]
        return seqs 

    def _postprocess(self, outputs:torch.FloatTensor, seqs:List[str]=None):
        ''''''
        # Transferring the outputs to CPU and reassigning should free up space on the GPU. 
        # https://discuss.pytorch.org/t/is-the-cuda-operation-performed-in-place/84961/6 
        if self.method in ['cls', 'gap']:
            outputs = outputs.last_hidden_state.cpu() # if (self.model_name == 'pt5') else outputs.pooler_output
            outputs = [self.pooler(emb, seq) for emb, seq in zip(outputs, seqs)]
        elif self.method in ['log']: 
            # Logits have shape (batch_size, seq_length, vocab_size), so this output should be a list of vocab_size tensors.
            # outputs = list(outputs.logits.cpu()[:, -1, :])
            outputs = [emb[len(seq) + 1] for emb, seq in zip(outputs, seqs)]
        return outputs        


def get_embedder(feature_type:str):
    ''' Instantiate the appropriate embedder for the specified feature type.'''
    if re.match('aa_([0-9]+)mer', feature_type) is not None:
        k = int(re.match('aa_([0-9]+)mer', feature_type).group(1))
        return KmerEmbedder(k=k)

    if feature_type == 'plm_pt5':
        return ProtT5Embedder()
    
    if re.match('plm_esm_(log|cls|gap)', feature_type) is not None:
        method = re.match('plm_esm_(log|cls|gap)', feature_type).group(1)
        return ESMEmbedder(method=method)

    if feature_type == 'len':
        return LengthEmbedder()

    return None


def embed(df:pd.DataFrame, path:str=None, overwrite:bool=False, feature_types:List[str]=None, format_='table'): 
    '''Embed the sequences in the input DataFrame (using all three embedding methods), and store the embeddings and metadata in an HDF5
    file at the specified path.'''

    store = pd.HDFStore(path, mode='a' if (not overwrite) else 'w') # Should confirm that the file already exists. 
    existing_keys = [key.replace('/', '') for key in store.keys()]
    df = df.sort_index() # Sort the index of the DataFrame to ensure consistent ordering. 

    mask = df.seq.isnull()
    if mask.sum() > 0:
        print(f'embed: Removing {np.sum(mask)} null entries from the sequence DataFrame. {len(df) - np.sum(seq_is_nan)} sequences remaining.', flush=True)
        df = df[~mask]

    if 'metadata' not in existing_keys:
        string_cols = [col for col in df.columns if (df[col].dtype == 'object')] 
        df[string_cols] = df[string_cols].fillna('none') # Avoid mixed column data types. 
        store.put('metadata', df, format=format_, data_columns=None)

    for feature_type in feature_types:
        if (feature_type in existing_keys) and (not overwrite):
            print(f'Embeddings of type {feature_type} are already present in {path}')
        else:
            print(f'embed: Generating embeddings for {feature_type}.')
            embedder = get_embedder(feature_type)
            embs, ids = embedder(df.seq.values.tolist(), df.index.values.tolist())
            sort_idxs = np.argsort(ids)
            embs, ids = embs[sort_idxs, :], ids[sort_idxs]
            emb_df = pd.DataFrame(embs, index=ids)
            store.put(feature_type, emb_df, format=format_, data_columns=None) 

    store.close()



