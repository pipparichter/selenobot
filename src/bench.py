'''
One of the questions which arose was, understandably, why dooes the linear classifier preform so well?
The suspicion is that it is simply picking up on amino acid composition, and that this is enough to distinguish
between truncated selenoproteins and short proteins, in this particular case. This "model" (which is not really a model)
is designed to test this theory.
'''
from torch.nn.functional import cross_entropy, binary_cross_entropy
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

# Tokenizer class to mimic the HuggingFace interface.   
# This class is used within SequenceDataset class. This class takes the entire DataFrame upon instantiation,
# and handles the labels this way. 
class BenchmarkTokenizer():
    def __init__(self): #, padding=False):
        '''

        '''

        aas = 'ARNDCQEGHILKMFPOSUTWYVBZXJ'
        self.map_ = {aas[i]: i + 1 for i in range(len(aas))} # Amino acid to integer map. 
        self.map_['<eos>'] = 0 # Add a termination token. 
        # self.padding = padding # Whether or not to zero-pad the data. 
        

    def __call__(self, data): # How to make the tokenizer callable. 
        '''
        args:
            - data (list): A list of amino acid sequences. 
        '''


        # Map each amino acid to an integer using self.map_
        mapped = [np.array([self.map_[res] for res in seq]) for seq in data]

        encoded_seqs = []
        # Create one-hot encoded array and sum along the rows (for each mapped sequence). 
        for mapped_seq in mapped:

            seq_length = len(mapped_seq)

            # Create an array of zeroes with the number of map elements as columns. 
            encoded_seq = np.zeros(shape=(seq_length, len(self.map_)))
            encoded_seq[np.arange(seq_length), mapped_seq] = 1
            encoded_seq = np.sum(encoded_seq, axis=0)
            # Now need to normalize according to sequence length. 
            encoded_seq = encoded_seq / seq_length

            encoded_seqs.append(encoded_seq)
        
        # if self.padding:
        #     max_length = len(max(encoded_seqs, key=len))
        #     for i, seq in enumerate(encoded_seqs):
        #         # Pad with zeros. 
        #         seq = np.concat(seq, np.zeros(max_length - len(seq)))
        #         encoded_seqs[i] = seq 

        return {'input_ids':np.array(encoded_seqs)}


# NOTE: Should I make this inheret from a torch.nn.module?
# class BenchmarkClassifier(torch.nn.Module):
# I think because there aren't weights being updated, etc. I should not do this. 
class BenchmarkClassifier():
    '''
    '''
    def __init__(self, n_components=2):

        # Want to produce a vector where each dimension corresponds to an amino acid, and the 
        # value contained represents the count. Also will probably want to normalize the vectors according to overall 
        # sequence length. 
        
        self.n_components = n_components # For the PCA model
        self.logreg = LogisticRegression() # (positive=True)

    def __call__(self, **kwargs):
        '''
        Just calls the predict function. 
        '''
        return self.predict(**kwargs)

    def fit(self, input_ids=None, labels=None):
        '''
        '''
        # pca = PCA(n_components=self.n_components)
        # pca_outputs = pca.fit_transform(input_ids)
        
        # NOTE: Why are the logreg outputs not 1 or 0?
        self.logreg.fit(input_ids, labels) # Should modify the object. 
        logreg_outputs = torch.DoubleTensor(self.logreg.predict(input_ids))
        logreg_probabilities = torch.DoubleTensor(self.logreg.predict_proba(input_ids)[:, 1])

        # Calculate loss if labels are specified. 
        loss = None
        if labels is not None:
            # Might need to convert things to tensors for this to work.  
            # loss = binary_cross_entropy(logreg_outputs, labels)
            loss = binary_cross_entropy(logreg_probabilities, labels)

        return logreg_outputs, loss

    # Just call the input data input_ids to match the PyTorch conventions. 
    def predict(self, input_ids=None, labels=None):
        '''

        kwargs:
            - input_ids (list): The tokenized input data.
            - labels (list)
        '''

        # Will want to instantiate a new PCA model each time.  
        # pca = PCA(n_components=self.n_components)
        # pca_outputs = pca.fit_transform(input_ids)

        # Pass the PCA outputs into the already-fitted linear regression thing. 
        logreg_outputs = torch.DoubleTensor(self.logreg.predict(input_ids))
        logreg_probabilities = torch.DoubleTensor(self.logreg.predict_proba(input_ids)[:, 1])
        
        # Calculate loss if labels are specified. 
        loss = None
        # NOTE: For loss, maybe I should be outputting specific probabilities, not labels. 
        if labels is not None:
            # Might need to convert things to tensors for this to work.  
            # loss = binary_cross_entropy(logreg_outputs, labels)
            loss = binary_cross_entropy(logreg_probabilities, labels)

        return logreg_outputs, loss


# NOTE: For this to make sense, the batch_sizes for the train and test loader must be the total number
# of datapoints (only one batch). 

def bench_train(model, train_loader):
    '''
    Fits the underlying linear regression-based benchmark model to the data. 

    args:
        - model (BenchmarkClassifier)
        - train_loader (DataLoader)
    '''
    for batch in train_loader:
        # This should pass in both the labels and the input_ids.
        predictions, loss = model.fit(**batch)
        accuracy = (predictions == batch['labels']).float().mean()
        break # Should only be one batch. Maybe a more user-friendly way to do this. 


    # If labels are given, go ahead and calculate loss, accuracy, etc. 
    return loss.item(), accuracy.item()


def bench_test(model, test_loader):
    '''
    Evaluate the benchmark model on the test data. 

    args:
        - model (BenchmarkClassifier)
        - test_loader (DataLoader)
    '''

    for batch in test_loader: # I think batch is a dictionary?
        predictions, loss = model(**batch)
        accuracy = (predictions == batch['labels']).float().mean()
        break # Should not go through the loop more than once. 

    return loss.item(), accuracy.item()



if __name__ == '__main__':
    pass 
    # from dataset import SequenceDataset
    # from torch.utils.data import DataLoader

    # tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    # kwargs = {'padding':True, 'truncation':True, 'return_tensors':'pt'}

    # # Grab the pre-loaded embeddings. 
    # train_embeddings = pd.read_csv('/home/prichter/Documents/protex/data/train_embeddings.csv')

    # train_data = SequenceDataset(pd.read_csv('/home/prichter/Documents/protex/data/train.csv'), tokenizer=tokenizer, embeddings=train_embeddings, **kwargs)
    # train_loader = DataLoader(train_data, batch_size=64) # Reasonable batch size?

    # # test_data = SequenceDataset(pd.read_csv('/home/prichter/Documents/protex/data/test.csv'), tokenizer=tokenizer, **kwargs)
    # # test_loader = DataLoader(test_data, batch_size=64) # Reasonable batch size?
    
    # model = ESMClassifierV2()
    # loss = esm_train(model, train_loader, n_epochs=200)
    # print(loss)
    
    # torch.save(model, '/home/prichter/Documents/protex/model_esm_v2.pickle')

