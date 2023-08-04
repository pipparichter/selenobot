'''
A generic linear classifier which sorts embedded amino acid sequences into two categories:
selenoprotein or non-selenoprotein. 
'''
import transformers
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# What functionality will span all classifier objects? Possibly put the 
# test and train functions here?
class Classifier(torch.nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()

        # Make sure everything is running on GPU. 
        self.to(device)

    def forward(self):
        '''
        This function should be overwritten in classes which inherit from this one. 
        '''
        # TODO: Probably shoulprint(dir(pr5_test_dataset))d have an error message pop up if this is ever called. 
        return None, None

    def test_(self, dataloader):
        '''
        '''
        self.eval() # Put the model in evaluation mode. 

        losses = []

        for batch in dataloader: 
            batch = {key:torch.Tensor(batch[key]) for key in ['data', 'label']}
            logits, loss = self(**batch)
        
            losses.append(loss.item()) # Add losses to a history. 

        return np.mean(losses) # Return the loss averaged over each batch. 

    def train_(self, dataloader, test_dataloader=None, epochs=300):
        '''
        '''
        self.train() # Put the model in train mode. 

        losses = {'train':[], 'test':[]}

        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        # This is trying to call train recursively...
        # self.train() # Put the model in training mode. 

        for epoch in tqdm(range(epochs)):

            for batch in dataloader:
                batch = {key:torch.Tensor(batch[key]) for key in ['data', 'label']}
                logits, loss = self(**batch)
                loss.backward() #
                optimizer.step() 
                optimizer.zero_grad()
            
            losses['train'].append(loss.item()) # Add losses to a history. 
            
            if test_dataloader is not None:
                test_loss = self.test_(test_dataloader)
                losses['test'].append(test_loss)
                # Make sure to put the model back in train mode. 
                self.train() 
        
        return losses


class NextTokenClassifier(Classifier):
    '''
    '''
    def __init__(self):

        super(NextTokenClassifierV1, self).__init__()
        self.gpt = transformers.GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2')

    def forward(self, **kwargs):
        return self._v1(**kwargs)

    def _v3(self, input=None, labels=None):
        pass
    
    def _v2(self, input=None, labels=None):
        pass

    def _v1(self, input=None, labels=None):
        '''
        '''
        # NOTE: Not sure if looping like this makes it way more inefficient. 

        preds = []
        # for id_, mask in zip(input_ids, attention_mask):
        for id_, mask, label in zip(input_ids, attention_mask, labels):
            id_ = id_[mask.to(torch.bool)]
            logits = self.gpt(id_).logits[-1] # Grab the logit for the last sequence element. 

            idxs = torch.argsort(logits, descending=True).numpy()
            # The 199 index represents the newline character, which should be discarded. 
            # See https://huggingface.co/nferruz/ProtGPT2/discussions/20. 
            next_token_idx = idxs[0] if (idxs[0] != 199) else idxs[1]

            print(label, self.tokenizer.decode(next_token_idx))

            if next_token_idx == 0: # This is the index for the end-of-text character. 
                preds.append(0)
            else:
                preds.append(1)

        # Putting the entire batch through the model at once was taking over 30 seconds per 10-element
        # batch, and was making my laptop tweak out. Accuracy was also weirdly low. Feeding in sequences
        # without all the padding, one-by-one, seems to be much faster. Not sure why.  

        # # Logits will have dims (batch_size, sequence_length, vocab_size)
        # # The -1 indexing grabs the last element of the sequence. 
        # logits = self.gpt(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        # idxs = torch.argsort(logits, dim=-1, descending=True)
        # idxs = ProtGPTClassifier._filter_newlines(idxs) # Remove the newline characters from consideration. 
        # # Seems as though the EOS character is indicated by zero. Newline is \n, which should be ignored. 
        # preds = torch.Tensor([0 if (i == 0) else 1 for i in idxs])
        # print(preds)

        return torch.Tensor(preds)



class EmbeddingClassifier(Classifier):
    '''
    '''
    def __init__(self, latent_dim):
        '''

        '''

        # Initialize the super class, torch.nn.Module. 
        super(EmbeddingClassifier, self).__init__()

        self.classifier = torch.nn.Linear(latent_dim, 1)


    def forward(self, data=None, label=None, **kwargs):
        '''
        A forward pass of the EmbeddingClassifier. In this case, the data 
        passed into the function should be sequence embeddings. 
        '''
        # Make sure the data is of the same type as the layer weights. 
        data = data.type(self.classifier.weight.dtype)

        logits = self.classifier(data)
        logits = torch.nn.functional.sigmoid(logits)
        
        loss = None
        if label is not None:
            loss = torch.nn.functional.binary_cross_entropy(torch.reshape(logits, label.size()), label.to(logits.dtype))
        
        return logits, loss
 

if __name__ == '__main__':

    from datasets import *
    from torch.utils.data import DataLoader

    figure_dir = '/home/prichter/Documents/selenobot/figures/'
    data_dir = '/home/prichter/Documents/selenobot/data/'

    train_data = pd.read_csv(data_dir + 'train.csv')
    test_data = pd.read_csv(data_dir + 'test.csv')
 
    dataset = EmbeddingDataset.from_csv(data_dir + 'train_embeddings_pr5.csv')

    dataloader = DataLoader(dataset, batch_size=32)

    # TODO: Probably should just allow you to specify a hidden dimension. 
    classifier = EmbeddingClassifier(dataset.latent_dim)
    classifier.train(dataloader)




