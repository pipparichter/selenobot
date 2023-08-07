'''
This file contains definitions for different Datasets, all of which inherit from the 
torch.utils.data.Dataset class (and are therefore compatible with a Dataloader)
'''

import pandas as pd
import numpy as np
import transformers
import torch
import h5py
from tqdm import tqdm

# I should probably think about a consistent format for all the embeddings. 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, metadata=None):

        self.data = data
        self.metadata = metadata

        # TODO: Make this adjustable. 
        # self.return_type = 'pt'

        self.length = len(data)

    def __repr__(self):

        pass

    def __getitem__(self, idx):

        item = {'data':self.data[idx]}
        
        if self.metadata is not None:
            for key, val in self.metadata.items():
                item[key] = val[idx]

        return item


    def to_csv(self, filename):

        # If the Dataset is not an EmbeddingDataset, then the data
        # should just be the sequence information. 
        df = self.metadata
        df['seq'] = self.data
        df = pd.DataFrame(df) # Convert the dictionary to a DataFrame. 

        df = df.set_index('id')
        df.to_csv(filename, index=True, index_label='id')

    def __len__(self):
        return self.length

# -----------------------------------------------------------------------------

class EmbeddingDataset(Dataset):

    def __init__(self, data, metadata=None, embedder=None, tokenizer=None):

        data = self.embed(data, embedder=embedder, tokenizer=tokenizer)

        super(EmbeddingDataset, self).__init__(data, metadata=metadata)

        self.latent_dim = self.data.shape[-1] # Get the size of the latent space. 


    def embed(self, data, embedder=None, tokenizer=None):
        '''
        '''
        # If no embedder is specified, don't do anything to the data. 
        if embedder is None:
            return data

        batch_size = 10
        n_batches = (len(data) // batch_size) + 1
        # Get indices for batch elements.  
        batches = np.array_split(np.arange(len(data)), n_batches)
        
        if tokenizer is not None:
            # Output of this is a dictionary, with keys relevant to the particular model. 
            data = tokenizer(list(data), return_tensors='pt', padding=True, truncation=True) 
        else:
            data = {'seqs':data}

        embeddings = []
        for batch in tqdm(batches):
            # Extract the batch data, and spin up on to a GPU, if available. 
            batch_data = {key:value[batch].to(device) for key, value in data.items()}

            embeddings.append(embedder(**batch_data))

        # Return the embedded data, which will be passed into the parent class constructor. 
        return np.concatenate(embeddings)

    # def from_h5(filename, metadata=None):
    #     '''
    #     Load an EmbeddingDataset object from a .h5 file.
    #     '''

    #     file_ = h5py.File(filename)
    #     data = [] 
    #     gene_names = file_.keys()

    #     for key in gene_names:
    #         # What does this do?
    #         data.append(file_[key][()])
        
    #     file_.close()

    #     data = np.array(data)

    #     return EmbeddingDataset(data, metadata=metadata)

    def from_csv(filename):
        
        # For now, expect columns id (which is the index), labels, and the embedding. 
        df = pd.read_csv(filename)
        
        # to_dict was being weird here, for whatever reason. 
        metadata = {}
        metadata['label'] = df['label'].values
        metadata['id'] = df['id'].values
        
        data = df.drop(columns=['label', 'id']).values

        return EmbeddingDataset(data, metadata=metadata)


    def to_csv(self, filename):
        '''
        Overwrites the to_csv method in the Dataset class. 
        '''
        df = pd.DataFrame(self.data)
        # Add all the metadata to the DataFrame.
        for key, value in self.metadata.items():
            df[key] = value 

        # I don't think it makes sense to write the sequences to the embedding file. 
        if 'seq' in df.columns:
            df = df.drop(columns=['seq']) 

        df = df.set_index('id')
        df.to_csv(filename, index=True, index_label='id')


class EsmEmbeddingDataset(EmbeddingDataset):

    # TODO: Look more into what setting attributes up here does. 

    model_name = 'facebook/esm2_t6_8M_UR50D'
    # model_name = 'esm2_t36_3B_UR50D'
    model = transformers.EsmModel.from_pretrained(model_name).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    def __init__(self, data, metadata=None, version=2):
        '''
        '''
        self.version = version 

        super(EsmEmbeddingDataset, self).__init__(data, metadata=metadata, embedder=self.embedder, tokenizer=EsmEmbeddingDataset.tokenizer)

    def embedder(self, input_ids=None, attention_mask=None):

        if self.version == 1:
            return EsmEmbeddingDataset._v1(input_ids=input_ids, attention_mask=attention_mask)
        elif self.version == 2:
            return EsmEmbeddingDataset._v2(input_ids=input_ids, attention_mask=attention_mask)
    
    def _v2(input_ids=None, attention_mask=None):
        '''
        Generates a sequence embedding using the CLS token. 
        '''
        EsmEmbeddingDataset.model.eval() # Put the model in evaluation mode. 
        # Extract the pooler output. 
        output = EsmEmbeddingDataset.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        # If I don't convert to a numpy array, the process crashes. 
        return output.detach().numpy()

    def _v1(input_ids=None, attention_mask=None):
        '''
        Generates a sequence embedding by mean-pooling over the sequence length. 
        '''
        EsmEmbeddingDataset.model.eval() # Put the model in evaluation mode. 

        output = EsmEmbeddingDataset.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
 
        # Sum the attention mask to get the length of each sequence. 
        denominator = torch.sum(attention_mask, -1, keepdim=True)
       
        # Get the dimensions of the attention mask to work. 
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)
        attention_mask = attention_mask.expand(-1, -1, 320)

        # Apply the attention mask to the output to effectively remove padding. 
        numerator = torch.mul(attention_mask, output)
        # Now sum up over the sequence length dimension. 
        numerator = torch.sum(numerator, dim=1) # Confirmed the shape is correct. 

        
        # EsmEmbeddingDataset.check_masking(numerator, output, attention_mask)
        return torch.divide(numerator, denominator).detach().numpy()
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
                

# TODO: Support different return types. 

class AacEmbeddingDataset(EmbeddingDataset):
    '''
    '''

    def __init__(self, data, metadata=None): #, padding=False):
        
        super(AacEmbeddingDataset, self).__init__(data, metadata=metadata, embedder=AacEmbeddingDataset.embedder)

    def embedder(seqs=None): 
        '''
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


# Testing!
if __name__ == '__main__':

    figure_dir = '/home/prichter/Documents/selenobot/figures/'
    data_dir = '/home/prichter/Documents/selenobot/data/'

    train_data = pd.read_csv(data_dir + 'train.csv')
    test_data = pd.read_csv(data_dir + 'test.csv')
 
    # data = AacEmbeddingDataset(train_data['seq'].values)
    seqs = train_data['seq'].values
    metadata = train_data.drop(columns=['seq']).to_dict()
    # data = EsmEmbeddingDataset(seqs, metadata=metadata)

    esm_train_dataset = EsmEmbeddingDataset(train_data['seq'].values, metadata={'id':train_data['id'].values, 'label':train_data['label'].values}, version=1)
    # esm_train_dataset.to_csv(data_dir + 'train_embeddings_esm_v2.csv')
    
    esm_test_dataset = EsmEmbeddingDataset(test_data['seq'].values, metadata={'id':test_data['id'].values, 'label':test_data['label'].values})
    # esm_test_dataset.to_csv(data_dir + 'test_embeddings_esm_v2.csv')



