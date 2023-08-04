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

        df.to_csv(filename, index=True, index_labels=list(self.metadata.keys()))

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
        for batch in batches:
            # Extract the batch data, and spin up on to a GPU, if available. 
            batch_data = {key:value[batch].to(device) for key, value in data.items()}

            embeddings.append(embedder(**batch_data))

        # Return the embedded data, which will be passed into the parent class constructor. 
        return np.concat(embeddings)

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
        df = df.drop(columns=['seq']) 

        df.to_csv(filename, index=True, index_labels=list(self.metadata.keys()))


class EsmEmbeddingDataset(EmbeddingDataset):

    # TODO: Look more into what setting attributes up here does. 

    model_name = 'facebook/esm2_t6_8M_UR50D'
    model = transformers.EsmModel.from_pretrained(model_name).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    def __init__(self, data, metadata=None):
        '''
        '''
        super(EsmEmbeddingDataset, self).__init__(data, metadata=metadata, embedder=EsmEmbeddingDataset.embedder, tokenizer=EsmEmbeddingDataset.tokenizer)

    def embedder(input_ids=None, attention_mask=None):
        
        EsmEmbeddingDataset.model.eval() # Put the model in evaluation mode. 

        output = EsmEmbeddingDataset.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # If I don't convert to a numpy array, the process crashes. 
        return torch.mean(output, 1).detach().numpy()
 

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

    dataset = EmbeddingDataset.from_csv(data_dir + 'test_embeddings_pr5.csv')




