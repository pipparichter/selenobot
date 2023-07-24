'''
Code for a baseline approach to classifying amino acid sequences. This approach treats each sequence like a "bag of words." 
'''
# import tensorflow as tf
import torch
import transformers
from torch.utils.data import DataLoader
# from transformers import AdamW
from torch.optim import Adam
from tqdm import tqdm
from torch.nn.functional import cross_entropy, binary_cross_entropy

from dataset import SequenceDataset
import numpy as np
import pandas as pd

# I think it would be acceptable to use the ESM Tokenizer here? Might be worth visualizing what exactly it's 
# doing first

class LinearClassifier(torch.nn.Module):
    '''
    Model which consists of a simple lookup table-based embedding layer, followed by logistic regression-based
    classification. This ignores sequential information when classifying sequences, treating each sequence as
    a "bag of words."
    '''

    def __init__(self, vocab_size=33, embedding_dim=64, use_embedding_layer=True): # , num_labels=1):
        '''
        '''
        # Initialize the super class. 
        super(LinearClassifier, self).__init__()

        self.use_embedding_layer = use_embedding_layer

        # Option to specify a padding index, but might not help here.
        # Also a max_norm option, which I don't think is necessary with the sigmoid activation. 
        self.embedder = torch.nn.EmbeddingBag(vocab_size, embedding_dim, mode='mean')

        # TODO: Maybe add some kind of intermediate layer?
        self.classifier = torch.nn.Linear(embedding_dim, 1)

        # https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are/13362#13362
        # Initialize the weights using Xavier uniform distribution. I think the biases are initialized to zero. 
        # NOTE: Maybe double-chack that this works inplace. 
        # torch.nn.init.xavier_normal_(self.classifier.weight)
        #torch.nn.init.xavier_normal_(self.embedder.weight)

    # This might be bad practice, but basically throwing out the attention mask input. 
    def forward(self, input_ids=None, labels=None, attention_mask=None):
        '''
        '''
        # Make sure the datatypes match; I was having issues with this. 

        # NOTE: I think this works for batch normalization?
        if self.use_embedding_layer:
            embedding = self.embedder(input_ids.to(torch.long))
            embedding = torch.nn.functional.sigmoid(embedding)
        else:
            embedding = input_ids.to(self.classifier.weight.dtype)
        
        # Then apply a classifier to the embedding. 
        logits = self.classifier(embedding)
        # When softmax is used, everything becomes one. I think we want sigmoid here. 
        # Also, every tutorial I saw used sigmoid. 
        logits = torch.nn.functional.sigmoid(logits)

        loss = None
        if labels is not None: # If labels are specified...
            # Make sure the labels match the datatype of the logits. Probably safer to change the label type. 
            labels = labels.to(logits.dtype)
            loss = binary_cross_entropy(torch.reshape(logits, labels.shape), labels)
        
        return logits, loss


# TODO: This is code duplication. Probably should come up with a way to organize functions. 
# Also kind of reluctant to put this in utils, because that's mostly file reading and writing.  
def linear_train(model, train_loader, test_loader=None, n_epochs=300):
    '''
    '''
    losses = {'train':[], 'test':[]}

    optimizer = Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train() # Put the model in training mode. 

    for epoch in tqdm(range(n_epochs), desc='Training classifier...'):
        
        for batch in train_loader:

            logits, loss = model(**batch)
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
        
        losses['train'].append(loss.item()) # Add losses to a history. 
        
        if test_loader is not None:
            test_loss, _ = linear_test(model, test_loader)
            losses['test'].append(test_loss.item())

    return losses


# TODO: Fix how labels are managed and stored. 
def linear_test(model, test_loader):
    '''
    Evaluate the model on the test data. 
    '''
    # model = model.to(device) # Make sure everything is running on the GPU. 
    
    model.eval() # Put the model in evaluation mode. 

    accuracy, loss = [], [] # Need to do this in batches of one.
    for batch in tqdm(test_loader, desc='Calculating batch loss...'): 
        # batch = {k: v.to(device) for k, v in batch.items()} # Mount on the GPU. 
        
        with torch.no_grad():
            batch_logits, batch_loss = model(**batch)
        loss.append(batch_loss.expand(1))

        batch_prediction = torch.round(batch_logits) # .to(device) # To zero or one.
        batch_accuracy = (batch_prediction == batch['labels']).float().mean() # Should be able to do this if I don't shuffle. 
        accuracy.append(batch_accuracy)

    # Concatenate the accumulated results. 
    loss = torch.mean(torch.cat(loss))
    accuracy = np.mean(accuracy)

    return loss.item(), accuracy.item()
    # return loss


# For testing... 
if __name__ == '__main__':

    from dataset import SequenceDataset
    from bench import BenchmarkTokenizer
    from torch.utils.data import DataLoader

    train_data = SequenceDataset(pd.read_csv('./data/train.csv'), tokenizer=BenchmarkTokenizer())
    test_data = SequenceDataset(pd.read_csv('./data/test.csv'), tokenizer=BenchmarkTokenizer())

    # Why is shuffling helpful? I feel like this is something I should know, but I don't. 
    train_loader = DataLoader(train_data, shuffle=False, batch_size=len(train_data))
    test_loader = DataLoader(train_data, shuffle=False, batch_size=len(test_data))

    model = LinearClassifier(embedding_dim=27, use_embedding_layer=True)
    
    # Loss is really high. After adjusting to use probabilities (rather than classification), this improved. 
    print(linear_train(model, train_loader))
    print(linear_test(model, test_loader))

