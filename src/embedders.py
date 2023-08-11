'''
This file contains a series of classes which allow for the creation of embedding datasets
from sequence data. They produce output files which are compatable with the 
Dataset defined in datasets.py.
'''
import transformers
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Embedder():

    def __init__(self, dataset):
        '''Initializes an Embedder object.'''

        # These should be populated when the inheriting class is called. 
        self.tokenizer = None

    def embedder(self, input_ids=None, attention_mask=None):
        '''This is overwritten in inheriting classes. All embedders should
        return np.arrays of embedded sequences.'''

        return None

    def embed(self, dataset, batch_size=1):
        '''Embeds the input data and returns an EmbeddingDataset object. 

        args:
            - dataset (SequenceDataset): A dataset containing the amino
                acid sequences for embedding. 
            - batch_size (int): Size of batches for processing the 
                embeddings. 
        '''
        n_batches = (len(dataset) // batch_size) + 1
        # Get indices for batch elements.  
        batches = np.array_split(np.arange(len(data)), n_batches)
        
        if tokenizer is not None:
            # Output of this is a dictionary, with keys relevant to the particular model. 
            data = tokenizer(list(data), return_tensors='pt', padding=True, truncation=True) 
        else: 
            # If no tokenizer is specified, just convert to a dictionary. 
            data = {'seqs':data}

        embeddings = []
        for batch in tqdm(batches):
            # Extract the batch data, and spin up on to a GPU, if available. 
            batch_data = {key:value[batch].to(device) for key, value in data.items()}
            embeddings.append(embedder(**batch_data))

        # Return the embedded data, which will be passed into the parent class constructor. 
        # return np.concatenate(embeddings)


class EsmEmbedder(Embedder):

    def __init__(self, dataset, pooling='cls', model_name='facebook/esm2_t6_8M_UR50D'):
        '''Initializes an EsmEmbedder object. 

        args:
            - pooling (str): The pooling method to use. Either 'cls' or 'mean'.
            - model_name (str): The pre-trained model to load from HuggingFace.
        '''
        # Make sure to initialize the parent class, so things don't get reset to None.
        super(Embedder, self).__init__()
        
        self.pooling = pooling

        # Probably should avoid storing these things in the object. 
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.EsmModel.from_pretrained(model_name).to(device)

    def embedder(self, input_ids=None, attention_mask=None):
        '''Overwrites the embedder method in the Embedder parent class. Generates an embedding
        using a pre-trained ESM model.  
        
        args:
            - input_ids (torch.Tensor): The tokenized sequence data. 
            - attention_mask (torch.Tensor): A masking array which tells
                the model which sequence elements are special tokens. 
        '''
        if self.pooling == 'cls':
            return self._cls(input_ids=input_ids, attention_mask=attention_mask)
        elif self.version == 'mean':
            return self._mean(input_ids=input_ids, attention_mask=attention_mask)
    
    def _cls(self, input_ids=None, attention_mask=None):
        '''Generates a sequence embedding using the CLS embedding for pooling. 

        args:
            - input_ids (torch.Tensor): The tokenized sequence data. 
            - attention_mask (torch.Tensor): A masking array which tells
                the model which sequence elements are special tokens. 
        '''
        self.model.eval() # Put the model in evaluation mode. 
        # Extract the pooler output. 
        output = self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        # If I don't convert to a numpy array, the process crashes. 
        return output.cpu().detach().numpy()

    def _mean(self, input_ids=None, attention_mask=None):
        '''Generates a sequence embedding by mean-pooling over the sequence length. 
        
        args:
            - input_ids (torch.Tensor): The tokenized sequence data. 
            - attention_mask (torch.Tensor): A masking array which tells
                the model which sequence elements are special tokens. 
        '''

        EsmEmbeddingDataset.model.eval() # Put the model in evaluation mode. 

        output = EsmEmbeddingDataset.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
 
        # Sum the attention mask to get the length of each sequence. 
        denominator = torch.sum(attention_mask, -1, keepdim=True)
       
        # Get the dimensions of the attention mask to work. 
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        attention_mask = attention_mask.expand(-1, -1, self.latent_dim)

        # Apply the attention mask to the output to effectively remove padding. 
        numerator = torch.mul(attention_mask, output)
        # Now sum up over the sequence length dimension. 
        numerator = torch.sum(numerator, dim=1) # Confirmed the shape is correct. 

        
        # EsmEmbeddingDataset.check_masking(numerator, output, attention_mask)
        return torch.divide(numerator, denominator).cpu().detach().numpy()
        # I checked to make sure this was doing what I thought. 
        

    # def check_masking(X, output, attention_mask):

    #     # Pobably easiest thing to do here is ravel and iterate. 
    #     # All three arguments should have the same dimension at this point. 
    #     if not ((X.shape == output.shape) and (X.shape == attention_mask.shape)):
    #         raise Exception('Dimensions of ESM model output, attention mask, and result of attention mask application do not match.')

    #     # Maybe a cleaner way to do this. Easiest thing is probably flatten and iterate over 1D tensors. 
    #     X, output, attention_mask = torch.flatten(X), torch.flatten(output), torch.flatten(attention_mask)
    #     for x, o, a in zip(X, output, attention_mask):
    #         if (a == 0) and (x != 0):
    #             raise Exception('Attention mask and numerator elements are mismatched.')
    #         if (a == 1) and (x != o):
    #             raise Exception('Output and numerator elements are mismatched.')
                


class AacEmbedder(Embedder):

    def __init__(self): 
        '''Initializes an AacEmbedder object.'''
        # There is no tokenizer in this case, so leave it as the default None. 
        super(AacEmbedder, self).__init__()

    def embedder(seqs=None): 
        '''Overwrites the embedder method in the Embedder parent class. Generates an embedding
        of a sequence based on amino acid content. 

        args:
            - seqs (list): A list of amino acid sequences. 
        '''
        aas = 'ARNDCQEGHILKMFPOSUTWYVBZXJ'
        aa_to_int_map = {aas[i]: i + 1 for i in range(len(aas))}
        aa_to_int_map['<eos>'] = 0 # Add a termination token. 

        
        # Map each amino acid to an integer using the internal map. 
        mapped = [np.array([aa_to_int_map[aa] for aa in seq]) for seq in seqs]

        embeddings = []
        # Create one-hot encoded array and sum along the rows (for each mapped sequence). 
        for seq in mapped:

            # Create an array of zeroes with the number of map elements as columns. 
            # This will be the newly-embeddings sequence. 
            e = np.zeros(shape=(len(seq), len(aa_to_int_map)))
            
            e[np.arange(len(seq)), seq] = 1
            e = np.sum(e, axis=0)
            # Now need to normalize according to sequence length. 
            e = e / len(seq)

            embeddings.append(e.tolist())
       
        # encoded_seqs is now a 2D list. Convert to numpy array for storage. 
        return np.array(embeddings)



