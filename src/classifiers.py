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
import torch.nn.functional
import torcheval
# from torcheval.metrics.functional import binary_accuracy
# import torchmetrics
from sklearn.metrics import balanced_accuracy_score
from reporter import Reporter


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weight=1):

        super(WeightedBCELoss, self).__init__()
        self.w = weight

    def forward(self, outputs, targets):
        '''Update the internal states keeping track of the loss.'''
        # reduction specifies the reduction to apply to the output. If 'none', no reduction will
        #  be applied, if 'mean,' the weighted mean of the output is taken.
        ce = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction='none')
        # Seems to be generating a weight vector, so that the weight is applied to indices marked with a 1. This
        # should have the effect of increasing the cost of a false negative. 
        w = torch.where(targets == 1, self.w, 1).to(device)

        return (ce * w).mean()


# Why use the torchmetrics package instead of the usual?
class Classifier(torch.nn.Module):

    def __init__(self, weight=1):

        super(Classifier, self).__init__()

        self.to(device)

    def forward(self):
        '''This function should be overwritten in classes which inherit from this one.'''
        pass 

    def init_weights(self):
        '''This function should be overwritten in classes which inherit from this one.'''
        pass 

    def reset(self):
        '''Reset the model weights according to the weight initialization function.'''
        self.init_weights()

    def predict(self, dataloader, threshold=None):
        '''Applies the model to a DataLoader, accumulating predictions across batches.
        Returns the model predictions, as well as '''
        self.eval() # Put the model in evaluation mode. 

        outputs, targets = [], []
        for batch in dataloader:
            batch_outputs, batch_targets = self(**batch)   
            outputs.append(batch_outputs)
            targets.append(batch_targets)

        # Concatenate outputs and labels into single tensors. 
        outputs, targets = torch.concat(outputs), torch.concat(targets)

        if threshold is not None:
            outputs = torch.where(outputs > threshold, 1, 0)
        
        self.train() # Go ahead and put back in train mode. 

        return outputs, targets


    def evaluate(self, dataloader, loss_func=None):
        '''Evaluate the performance of the model on the input dataloader. Returns the loss
        of the model on the data, as well as the accuracy.'''

        assert loss_func is not None, 'classifiers.Classifier.evaluate: A loss function must be specified.'
        outputs, targets = self.predict(dataloader)

        loss = loss_func(outputs, targets)
        accuracy = balanced_accuracy_score(outputs, targets, threshold=0.5)

        return loss, accuracy


    def _train(self, dataloader, val=None, epochs=300, lr=0.01, bce_loss_weight=1):

        # Instantiate the loss function object with the specified weight. 
        loss_func = WeightedBCELoss(weight=bce_loss_weight)

        self.train() # Put the model in train mode. 

        reporter = Reporter(epochs=epochs, lr=lr, bce_loss_weight=bce_loss_weight)
        reporter.set_batches_per_epoch(len(list(dataloader)))
        reporter.open()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in pbar:

            # If a validation dataset is specified, evaliate. 
            # NOTE: If I try to do this with every batch, it takes forever.  
            if val is not None:
                val_loss, val_acc = self.evaluate(val, loss_func=loss_func)
                reporter.add_val_metrics(val_loss, val_acc)

            for batch in dataloader:

                # Evaluate the model on the batch in the training dataloader. 
                outputs, targets = self(**batch)
                loss = loss_func(outputs, targets)
                acc = balanced_accuracy_score(outputs, targets)

                reporter.add_train_metrics(train_loss, train_acc)

                loss.backward()
                optimizer.step() 
                optimizer.zero_grad()

        reporter.close()
        
        return reporter

    def save(self, path):
        '''Save the model weights'''
        torch.save(self.state_dict(), path)

    def load(self, path):
        '''Load model weights'''
        self.load_state_dict(torch.load(path))


class EmbeddingClassifier(Classifier):
    '''A classifier which works on embedding data.'''
    def __init__(self, latent_dim, dropout=0):
        # Latent dimension should be 1024. 
        hidden_dim = 512

        # Initialize the super class, torch.nn.Module. 
        super(EmbeddingClassifier, self).__init__()
        
        # Subbing in Josh's classification head. 
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid())

        self.init_weights()

    def init_weights(self):
        '''Initialize model weights according to which activation is used.'''
        torch.nn.init.kaiming_normal_(self.classifier[0].weight)
        torch.nn.init.xavier_normal_(self.classifier[3].weight)


    def forward(self, data=None, label=None, **kwargs):
        '''
        A forward pass of the EmbeddingClassifier. In this case, the data 
        passed into the function should be sequence embeddings. 
        '''
        logits = self.classifier(data)
        # logits = torch.nn.functional.sigmoid(logits)        
        loss = None

        if label is not None:
            # loss = torch.nn.functional.binary_cross_entropy(torch.reshape(logits, label.size()), label.to(logits.dtype))
            outputs = torch.reshape(logits, label.size())
            targets = label.to(logits.dtype)

        return outputs, targets


class AacClassifier(Classifier):
    '''The "stupid" approach to selenoprotein detection. Simply uses amino acid content 
    to predict whether or not something is a selenoprotein. This should not work well.'''

    def __init__(self, weight=1):
        
        super(AacClassifier, self).__init__(weight=weight)

        aas = 'ARNDCQEGHILKMFPOSUTWYVBZXJ'
        latent_dim = len(aas) # The AAC embedding space is 22(?)-dimensional. 
        self.aa_to_int_map = {aas[i]: i + 1 for i in range(len(aas))}

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 1),
            torch.nn.Sigmoid())

        self.init_weights()

    def init_weights(self):
        '''Initialize model weights according to which activation is used.'''
        torch.nn.init.xavier_normal_(self.classifier[0].weight)
 
    def tokenize(self, seq):
        '''Embed a single sequence using amino acid content.''' 

        # Map each amino acid to an integer using the internal map. 
        seq = np.array([self.aa_to_int_map[aa] for aa in seq])
        
        emb = np.zeros(shape=(len(seq), len(self.aa_to_int_map)))
        emb[np.arange(len(seq)), seq] = 1
        emb = np.sum(emb, axis=0)
        # Now need to normalize according to sequence length. 
        emb = emb / len(seq)

        # encoded_seqs is now a 2D list. Convert to numpy array for storage. 
        return list(emb)

    def forward(self, data=None, label=None, **kwargs):
        '''
        A forward pass of the EmbeddingClassifier. In this case, the data 
        passed into the function should be sequence embeddings. 
        '''
        # I think data is going to be a tensor of shape (batch_size), where each element is a sequence string. 
        data = torch.Tensor([[self.tokenize(seq) for seq in data]]).to(torch.float32)

        logits = self.classifier(data)

        if label is not None:
            # loss = torch.nn.functional.binary_cross_entropy(torch.reshape(logits, label.size()), label.to(logits.dtype))
            outputs = torch.reshape(logits, label.size())
            targets = label.to(logits.dtype)

        return outputs, targets



# class NextTokenClassifier(Classifier):
#     '''    '''
#     def __init__(self):

#         super(NextTokenClassifierV1, self).__init__()
#         self.gpt = transformers.GPT2LMHeadModel.from_pretrained('nferruz/ProtGPT2')

#     def forward(self, **kwargs):
#         return self._v1(**kwargs)

#     def _v3(self, input=None, labels=None):
#         pass
    
#     def _v2(self, input=None, labels=None):
#         pass

#     def _v1(self, input=None, labels=None):
#         '''
#         '''
#         # NOTE: Not sure if looping like this makes it way more inefficient. 

#         preds = []
#         # for id_, mask in zip(input_ids, attention_mask):
#         for id_, mask, label in zip(input_ids, attention_mask, labels):
#             id_ = id_[mask.to(torch.bool)]
#             logits = self.gpt(id_).logits[-1] # Grab the logit for the last sequence element. 

#             idxs = torch.argsort(logits, descending=True).numpy()
#             # The 199 index represents the newline character, which should be discarded. 
#             # See https://huggingface.co/nferruz/ProtGPT2/discussions/20. 
#             next_token_idx = idxs[0] if (idxs[0] != 199) else idxs[1]

#             # print(label, self.tokenizer.decode(next_token_idx))

#             if next_token_idx == 0: # This is the index for the end-of-text character. 
#                 preds.append(0)
#             else:
#                 preds.append(1)

#         # Putting the entire batch through the model at once was taking over 30 seconds per 10-element
#         # batch, and was making my laptop tweak out. Accuracy was also weirdly low. Feeding in sequences
#         # without all the padding, one-by-one, seems to be much faster. Not sure why.  

#         # # Logits will have dims (batch_size, sequence_length, vocab_size)
#         # # The -1 indexing grabs the last element of the sequence. 
#         # logits = self.gpt(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
#         # idxs = torch.argsort(logits, dim=-1, descending=True)
#         # idxs = ProtGPTClassifier._filter_newlines(idxs) # Remove the newline characters from consideration. 
#         # # Seems as though the EOS character is indicated by zero. Newline is \n, which should be ignored. 
#         # preds = torch.Tensor([0 if (i == 0) else 1 for i in idxs])
#         # print(preds)

#         return torch.Tensor(preds)


# class WeightedBinaryCrossEntropy(torchmetrics.Metric):
#     '''A custom loss function which adjusts the penalty of misclassifying
#         a positive target instance (i.e. those marked by a 1).'''

#     def __init__(self, weight=1):

#         super(WeightedBinaryCrossEntropy, self).__init__()
        
#         # dist_reduce_fx is the unction to reduce state across multiple processes in distributed mode.
#         # Not really sure what distributed mode is. 
 
#         self.add_state('total', torch.Tensor(0).to(device), dist_reduce_fx='sum')
#         self.add_state('n', torch.Tensor(0).to(device), dist_reduce_fx='sum')

#     def compute(self):
#         '''Compute the cumulative loss and return the value.'''
#         return self.total / self.n
        
#     def update(self, inputs, target):
#         '''Update the internal states keeping track of the loss.'''
#         # reduction specifies the reduction to apply to the output. If 'none', no reduction will
#         #  be applied, if 'mean,' the weighted mean of the output is taken.
#         ce = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction='none')
#         # Seems to be generating a weight vector, so that the weight is applied to indices marked with a 1. This
#         # should have the effect of increasing the cost of a false negative. 
#         weight = torch.where(targets == 1, self.weight, 1).to(device)
        
#         self.total += torch.sum((ce * weight))
#         self.n += inputs.numel()

 





