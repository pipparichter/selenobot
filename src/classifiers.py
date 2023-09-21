'''
A generic linear classifier which sorts embedded amino acid sequences into two categories:
selenoprotein or non-selenoprotein. 
'''

import sys
sys.path.append('/home/prichter/Documents/selenobot/src')

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
import sklearn.metrics
from reporter import Reporter
import skopt 

import warnings
warnings.simplefilter('ignore')


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


def apply_threshold(outputs:torch.Tensor, threshold:float=0.5) -> torch.Tensor:
    '''Apply a threshold to model outputs to convert them to binary classification.'''
    # If no threshold is specified, then just return the outputs. 
    assert type(threshold) == float, 'classifiers.apply_threshold: Specified threshold must be a float.'
    return  torch.where(outputs < threshold, 0, 1)


def get_balanced_accuracy(outputs:torch.Tensor, targets:torch.Tensor, threshold:float=0.5) -> float:
    '''Applies a threshold to the model outputs, and uses it to compute a balanced accuracy
    score.'''
    outputs = apply_threshold(outputs, threshold=threshold)
    # Compute balanced accuracy using a builtin sklearn function. 
    return sklearn.metrics.balanced_accuracy_score(outputs.detach().numpy(), targets.detach().numpy())


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

    def predict(self, dataloader:torch.utils.data.DataLoader) -> (torch.Tensor, torch.Tensor):
    # def predict(self, dataloader:torch.utils.data.DataLoader, threshold:float=0.5) -> (torch.Tensor, torch.Tensor):
        '''Applies the model to a DataLoader, accumulating predictions across batches.
        Returns the model predictions, as well as the labels of the data.'''
        self.eval() # Put the model in evaluation mode. 

        outputs, targets = [], []
        for batch in dataloader:
            batch_outputs, batch_targets = self(**batch)   
            outputs.append(batch_outputs)
            targets.append(batch_targets)

        # Concatenate outputs and labels into single tensors. 
        outputs, targets = torch.concat(outputs), torch.concat(targets)
        
        self.train() # Go ahead and put back in train mode. 

        # Make sure outputs and target have the same datatype. 
        return outputs.to(targets.dtype), targets

    def test(self, 
        dataloader:torch.utils.data.DataLoader, 
        bce_loss_weight:float=1.0, 
        threshold:float=0.5, 
        reporter:Reporter=None) -> Reporter:
        '''Evaluate the classifier model on the data specified by the DataLoader

        args:
            - dataloader: The DataLoader object containing the training data. 
            - bce_loss_weight: The weight to be passed into the WeightedBCELoss constructor.
            - threshold: The threshold above which (inclusive) a classification is considered a '1.' 
            - reporter: An existing reporter to which to add test information. 
        ''' 
        # Instantiate the loss function object with the specified weight. 
        loss_func = WeightedBCELoss(weight=bce_loss_weight)
        
        if reporter is None:
            reporter = Reporter() # Instantiate a Reporter for storing collected data. 
        reporter.open()

        # NOTE: threshold should be None here!
        outputs, targets = self.predict(dataloader)
        test_loss = loss_func(outputs, targets)
        test_acc = get_balanced_accuracy(outputs, targets, threshold=threshold)
        reporter.add_test_metrics(test_loss, test_acc)

        # Compute the confusion matrix information. 
        outputs = apply_threshold(outputs, threshold=threshold) # Make sure to apply threshold to output data first. 
        (tn, fp, fn, tp) = np.ravel(sklearn.metrics.confusion_matrix(outputs.detach().numpy(), targets.detach().numpy()))
        reporter.add_confusion_matrix(tn, fp, fn, tp)

        reporter.close()
        return reporter

    # NOTE: Is is appropriate to call a training model "fit"? Is there a distinction between training and fitting?
    def fit(self, 
        dataloader:torch.utils.data.DataLoader, 
        val:torch.utils.data.DataLoader=None, 
        epochs:int=300, 
        lr:float=0.01, 
        bce_loss_weight:float=1.0, 
        threshold:float=0.5) -> Reporter:
        '''Train the classifier model on the data specified by the DataLoader

        args:
            - dataloader: The DataLoader object containing the training data. 
            - val: The DataLoader object containing the validation data. 
            - epochs: The number of epochs to train for. 
            - lr: The learning rate. 
            - bce_loss_weight: The weight to be passed into the WeightedBCELoss constructor.
            - threshold: The threshold above which (inclusive) a classification is considered a '1.' 
        '''

        # Instantiate the loss function object with the specified weight. 
        loss_func = WeightedBCELoss(weight=bce_loss_weight)

        self.train() # Put the model in train mode. 

        reporter = Reporter(epochs=epochs, lr=lr, bce_loss_weight=bce_loss_weight)
        reporter.set_batches_per_epoch(len(list(dataloader)))
        reporter.open()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        pbar = tqdm(range(epochs), desc='classifiers.Classifier.train_', leave=False) # disable=(not verbose))

        for epoch in pbar:

            # If a validation dataset is specified, evaliate. 
            # NOTE: If I try to do this with every batch, it takes forever.  
            if val is not None:
                # NOTE: Threshold should be None here. 
                outputs, targets = self.predict(val)

                val_loss = loss_func(outputs, targets)
                val_acc = get_balanced_accuracy(outputs, targets, threshold=threshold)
                reporter.add_val_metrics(val_loss, val_acc)
                pbar.set_postfix({'val_acc':np.round(val_acc, 2)})

            for batch in dataloader:

                # Evaluate the model on the batch in the training dataloader. 
                outputs, targets = self(**batch)
                train_loss = loss_func(outputs, targets)
                train_acc = get_balanced_accuracy(outputs, targets, threshold=threshold)
                reporter.add_train_metrics(train_loss, train_acc)

                train_loss.backward()
                optimizer.step() 
                optimizer.zero_grad()

            pbar.update()

        reporter.close()
        pbar.close()
        
        return reporter

    def save(self, path):
        '''Save the model weights'''
        torch.save(self.state_dict(), path)

    def load(self, path):
        '''Load model weights'''
        self.load_state_dict(torch.load(path))


class EmbeddingClassifier(Classifier):
    '''A classifier which works on embedding data.'''
    def __init__(self, latent_dim=1024, dropout=0):
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


    def forward(self, emb=None, label=None, **kwargs):
        '''
        A forward pass of the EmbeddingClassifier. In this case, the data 
        passed into the function should be sequence embeddings. 
        '''
        logits = self.classifier(emb)
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

    def forward(self, seq=None, label=None, **kwargs):
        '''
        A forward pass of the EmbeddingClassifier. In this case, the data 
        passed into the function should be sequence embeddings. 
        '''
        # I think data is going to be a tensor of shape (batch_size), where each element is a sequence string. 
        data = torch.Tensor([[self.tokenize(s) for s in seq]]).to(torch.float32)

        logits = self.classifier(data)

        if label is not None:
            # loss = torch.nn.functional.binary_cross_entropy(torch.reshape(logits, label.size()), label.to(logits.dtype))
            outputs = torch.reshape(logits, label.size())
            targets = label.to(logits.dtype)

        return outputs, targets



def optimize_hyperparameters(
        dataloader:torch.utils.data.DataLoader, 
        val:torch.utils.data.DataLoader, 
        model:Classifier=EmbeddingClassifier(), 
        n_calls:int=50,
        verbose:bool=True, 
        epochs:int=5) -> list: # -> skopt.OptimizeResult:

    # Probably keep the weights as integers, at least for now. 
    search_space = [skopt.space.Integer(1, 10000, name='bce_loss_weight')]

    # Create the objective function. Decorator allows named arguments to be inferred from search space. 
    @skopt.utils.use_named_args(dimensions=search_space)
    def objective(bce_loss_weight=None):
        # if verbose: print(f'classifiers.optimize_hyperparameters: Testing bce_loss_weight={bce_loss_weight}.')
        model.fit(dataloader, epochs=epochs, lr=0.01, bce_loss_weight=bce_loss_weight, threshold=0.5)
        
        # Evaluate the performance of the fitted model on the data.
        reporter = model.test(val, bce_loss_weight=bce_loss_weight, threshold=0.5)
        test_loss = reporter.get_test_losses()[0] # Get the test loss following evaluation on the data. 
        if verbose: print(f'classifiers.optimize_hyperparameters: Recorded a test loss of {np.round(test_loss, 2)} with bce_loss_weight={bce_loss_weight}.')
        return test_loss
    
    # We are using loss, so acceptable to just minimize the output of the objective function. 
    result = skopt.gp_minimize(objective, search_space)
    return result.x


def get_roc_info():
    pass




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

 





