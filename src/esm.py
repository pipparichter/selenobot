'''
This file contains code for producing embeddings of amino acid sequences using the pre-trained ESM
model (imported from HuggingFace)
'''
import transformers # Import the pretrained models from HuggingFace.
from transformers import EsmForSequenceClassification, EsmModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.optim import Adam
from torch.nn.functional import cross_entropy, binary_cross_entropy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ESMClassifierV2(torch.nn.Module):
    
    def __init__(self, name='facebook/esm2_t6_8M_UR50D'):
        '''
        Initializes an ESMClassifierV2 object, as well as the torch.nn.Module superclass. 
        
        args:
            - name (str): The name of the pretrained model to use. 
        '''
        # Initialize the super class, torch.nn.Module. 
        super(ESMClassifierV2, self).__init__()

        # I don't think there is a way to turn off the pooling layer when loading from pretrained. 
        # For now, just ignore it. 
        self.esm = EsmModel.from_pretrained(name)

        # Should be 320... 
        hidden_size = self.esm.get_submodule('encoder.layer.5.output.dense').out_features

        # TODO Possibly add a couple of layers, because I think hidden_size is large.
        self.classifier = torch.nn.Linear(hidden_size, 1)

        # Freeze all esm model weights. 
        for name, param in self.esm.named_parameters():
            param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, index=None, embeddings=None):
        '''

        args:
            - input_ids (torch.Tensor): Has a shape of (batch_size, sequence length). I think these are just the tokenized
                equivalents of each sequence element. 
            - attention_mask (torch.Tensor): A tensor of ones and zeros. 
            - labels (torch.Tensor): Has a size of (batch_size,). Indicated whether or not a sequence is truncated (1)
                or full-length (0).
            - index (int): The indices corresponding to the batch data's original position in the dataset. 
            - embeddings (np.array): The ESM-generated embedding of the amino acid sequence. 
        '''
        if embeddings is not None:
            # Make sure data types line up, so the linear layer doesn't flip out. 
            embedding = embeddings.to(self.classifier.weight.dtype)
        else: # Only bother doing a forward pass if the embedding is not already generated. 
            # Extract the weights from the model.
            last_hidden_state = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # Decided on sigmoid activation, not sure if something else is better. 
            last_hidden_state = torch.nn.functional.sigmoid(last_hidden_state)
            embedding = torch.mean(last_hidden_state, 1)
            # Will be useful to write the encodings to a file. Should be a sequence of 320-dimensional vectors. 

        # NOTE: Do we want to take the average before or after nonlinear normalization?
        # Probably before?
        logits = self.classifier(embedding)
        logits = torch.nn.functional.sigmoid(logits)
        
        # CODE FOR WRITING ESM EMBEDDINGS ----------------------------------------------
        # # Easiest thing to do is probably convert from tensor to numpy to CSV. 
        # encoding = encoding.numpy()
        # encoding = pd.DataFrame(embedding) # Not sure what the column names will be here?
        # encoding.to_csv('/home/prichter/Documents/protex/data/test_embeddings.csv', index=False, header=False, mode='a')

        # indices = pd.DataFrame(index.numpy())
        # indices.to_csv('/home/prichter/Documents/protex/data/test_indices.csv', index=False, header=False, mode='a')
        # ------------------------------------------------------------------------------

        loss = None
        if labels is not None:
            loss = binary_cross_entropy(torch.reshape(logits, labels.size()), labels.to(logits.dtype))
        
        return logits, loss

class ESMClassifierV1(torch.nn.Module):
    
    def __init__(self, name='facebook/esm2_t6_8M_UR50D'):
        '''
        Initializes an ESMClassifier object, as well as the torch.nn.Module superclass. 
        '''
        # Initialize the super class, torch.nn.Module. 
        super(ESMClassifier, self).__init__()

        self.model = EsmForSequenceClassification.from_pretrained(name, num_labels=1)

        # Freeze all model weights which aren't related to the classifier. 
        for name, param in self.model.esm.named_parameters():
            param.requires_grad = False
   
    def forward(self, input_ids=None, attention_mask=None, labels=None, index=None, embeddings=None):
        '''

        kwargs:
            - input_ids (torch.Tensor): Has a shape of (batch_size, sequence length). I think these are just the tokenized
                equivalents of each sequence element. 
            - attention_mask (torch.Tensor): A tensor of ones and zeros. 
            - labels (torch.Tensor): Has a size of (batch_size,). Indicated whether or not a sequence is truncated (1)
                or full-length (0).
        
        returns: transformers.modeling_outputs.SequenceClassifierOutput
        '''
        # if self.use_builtin_classifier:
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
        # Normalize the predictions. 
        logits = torch.nn.functional.sigmoid(logits)

        loss = None
        if labels is not None:
            loss = binary_cross_entropy(logits, labels)
        
        return logits, loss


# TODO: This is code duplication. Probably should come up with a way to organize functions. 
# Also kind of reluctant to put this in utils, because that's mostly file reading and writing.  
def esm_train(model, train_loader, test_loader=None, n_epochs=300):
    '''

    args:
        - model (torch.nn.Module)
        - train_loader (torch.utils.data.DataLoader)
    '''
    model = model.to(device) # Make sure everything is running on the GPU. 

    # losses = {'train':[], 'test':[], 'accuracy':[]}
    losses = {'train':[], 'test':[]}

    optimizer = Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train() # Put the model in training mode. 

    for epoch in tqdm(range(n_epochs), desc='Training classifier...'):

        batch_count = 0
        batch_total = len(train_loader)

        for batch in train_loader:
            print(f'BATCH {batch_count}/{batch_total}\t', end='\r')
            batch_count += 1

            batch = {k: v.to(device) for k, v in batch.items()} # Mount on the GPU. 

            logits, loss = model(**batch)
            loss.backward() #
            optimizer.step() 
            optimizer.zero_grad()
        
        losses['train'].append(loss.item()) # Add losses to a history. 
        
        if test_loader is not None:
            # test_loss, accuracy = esm_test(model, test_loader)
            test_loss, _ = esm_test(model, test_loader)
            losses['test'].append(test_loss.item())
            # losses['accuracy'].append(accuracy.item())
            # Make sure to put the model back in train mode. 
            model.train()

    return losses


# TODO: Fix how labels are managed and stored. 
def esm_test(model, test_loader, embedding_file=None):
    '''
    Evaluate the model on the test data. 
    '''
    model = model.to(device) # Make sure everything is running on the GPU. 
    
    model.eval() # Put the model in evaluation mode. 

    accuracy, loss = [], []
    for batch in tqdm(test_loader, desc='Calculating batch loss...'): 
        batch = {k: v.to(device) for k, v in batch.items()} # Mount on the GPU. 
        
        with torch.no_grad():
            batch_logits, batch_loss = model(**batch)
        loss.append(batch_loss.expand(1))

        batch_prediction = torch.round(batch_logits) # .to(device) # To zero or one.
        # NOTE: I think it doesn't actually matter if I shuffle or not. 
        batch_accuracy = (batch_prediction == batch['labels']).float().mean() # Should be able to do this if I don't shuffle. 
        accuracy.append(batch_accuracy)

    # Concatenate the accumulated results. 
    loss = torch.mean(torch.cat(loss))
    accuracy = np.mean(accuracy)

    return loss, accuracy


if __name__ == '__main__':
    from dataset import SequenceDataset
    from torch.utils.data import DataLoader
    from transformers import EsmTokenizer


    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    kwargs = {'padding':True, 'truncation':True, 'return_tensors':'pt'}

    # Grab the pre-loaded embeddings. 
    train_embeddings = pd.read_csv('/home/prichter/Documents/protex/data/train_embeddings.csv')

    train_data = SequenceDataset(pd.read_csv('/home/prichter/Documents/protex/data/train.csv'), tokenizer=tokenizer, embeddings=train_embeddings, **kwargs)
    train_loader = DataLoader(train_data, batch_size=64) # Reasonable batch size?

    # test_data = SequenceDataset(pd.read_csv('/home/prichter/Documents/protex/data/test.csv'), tokenizer=tokenizer, **kwargs)
    # test_loader = DataLoader(test_data, batch_size=64) # Reasonable batch size?
    
    model = ESMClassifierV2()
    loss = esm_train(model, train_loader, n_epochs=200)
    print(loss)
    
    torch.save(model, '/home/prichter/Documents/protex/model_esm_v2.pickle')



