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
import data

# I think it would be acceptable to use the ESM Tokenizer here? Might be worth visualizing what exactly it's 
# doing first

class LogisticRegressionClassifier(torch.nn.Module):
    '''
    Model which consists of a simple lookup table-based embedding layer, followed by logistic regression-based
    classification. This ignores sequential information when classifying sequences, treating each sequence as
    a "bag of words."
    '''

    def __init__(self, vocab_size=33, embedding_dim=64): # , num_labels=1):
        '''
        '''
        # Initialize the super class. 
        super(LogisticRegressionClassifier, self).__init__()

        # Option to specify a padding index, but might not help here.
        # Also a max_norm option, which I don't think is necessary with the sigmoid activation. 
        # self.embedder = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedder = torch.nn.EmbeddingBag(vocab_size, embedding_dim, mode='mean')

        # TODO: Maybe add some kind of intermediate layer?
        self.classifier = torch.nn.Linear(embedding_dim, 1)

        # https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are/13362#13362
        # Initialize the weights using Xavier uniform distribution. I think the biases are initialized to zero. 
        # NOTE: Maybe double-chack that this works inplace. 
        torch.nn.init.xavier_normal_(self.classifier.weight)
        torch.nn.init.xavier_normal_(self.embedder.weight)

    # This might be bad practice, but basically throwing out the attention mask input. 
    def forward(self, input_ids=None, labels=None, attention_mask=None):
        '''
        '''
        embedding = self.embedder(input_ids)
        # NOTE: I think this works for batch normalization?
        embedding = torch.nn.functional.sigmoid(embedding)
        # Then apply a classifier to the embedding. 
        logits = self.classifier(embedding)
        # When softmax is used, everything becomes one. I think we want sigmoid here. 
        # Also, every tutorial I saw used sigmoid. 
        logits = torch.nn.functional.sigmoid(logits)

        loss = None
        if labels is not None: # If labels are specified...
            loss = binary_cross_entropy(logits, labels)
        
        return logits, loss


# TODO: This is code duplication. Probably should come up with a way to organize functions. 
# Also kind of reluctant to put this in utils, because that's mostly file reading and writing.  
def logreg_train(model, train_data, test_data=None, batch_size=10, shuffle=True, n_epochs=300):
    '''
    '''
    losses = {'train':[], 'test':[], 'accuracy':[]}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

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
        
        if test_data is not None:
            test_loss, accuracy = logreg_test(model, test_data)
            losses['test'].append(test_loss.item())
            losses['accuracy'].append(accuracy.item())

    return losses


def logreg_test(model, test_data):
    '''
    Evaluate the model on the test data. 
    '''
    test_loader = DataLoader(test_data, batch_size=len(test_data.labels), shuffle=False)
    
    model.eval() # Put the model in evaluation mode. 
    for batch in test_loader: # Should only be one batch. 
        
        with torch.no_grad():
            logits, loss = model(**batch)

    # In addition to the loss, get the accuracy. 
    prediction = torch.round(logits) # To zero or one. 
    accuracy = (prediction == test_data.labels).float().mean() # Should be able to do this if I don't shuffle. 

    return loss, accuracy


if __name__ == '__main__':

    model_v3 = LogisticRegressionClassifier()
    losses_v3 = logreg_train(model_v3, train_data, batch_size=100, n_epochs=100)
