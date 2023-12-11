'''A binary classification head and associated utilities, designed for handling embedding objects.'''
from selenobot.dataset import Dataset
from selenobot.reporters import TestReporter, TrainReporter

from tqdm import tqdm
from typing import Optional, NoReturn, Tuple, List

import sys
import torch
import os
import numpy as np
import pandas as pd
import torch.nn.functional
import sklearn.metrics
import skopt 
import warnings

warnings.simplefilter('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class WeightedBCELoss(torch.nn.Module):
    '''Defining a class for easily working with weighted Binary Cross Entropy loss.'''
    def __init__(self, weight=1):

        super(WeightedBCELoss, self).__init__()
        self.w = weight

    def forward(self, outputs, targets):
        '''Update the internal states keeping track of the loss.'''
        # Make sure the outputs and targets have the same shape.
        outputs = outputs.reshape(targets.shape)
        # reduction specifies the reduction to apply to the output. If 'none', no reduction will be applied, if 'mean,' the weighted mean of the output is taken.
        ce = torch.nn.functional.binary_cross_entropy(outputs, targets, reduction='none')
        # Seems to be generating a weight vector, so that the weight is applied to indices marked with a 1. This
        # should have the effect of increasing the cost of a false negative.
        w = torch.where(targets == 1, self.w, 1).to(DEVICE)

        return (ce * w).mean()


def get_balanced_accuracy(
    outputs:torch.FloatTensor, 
    targets:torch.FloatTensor, 
    threshold:float=0.5) -> float:
    '''Applies a threshold to the model outputs, and uses it to compute a balanced accuracy score.'''
    outputs = outputs.reshape(targets.shape)
    outputs = torch.where(outputs < threshold, 0, 1)
    # Compute balanced accuracy using a builtin sklearn function. 
    return sklearn.metrics.balanced_accuracy_score(targets.detach().numpy(), outputs.detach().numpy())


class Classifier(torch.nn.Module):
    '''Class defining the binary classification head.'''

    def __init__(self, 
        hidden_dim:int=512,
        latent_dim:int=1024,
        dropout:float=0, 
        bce_loss_weight:float=1.0):
        '''
        Initializes a two-layer linear classification head. 

        args:
            - bce_loss_weight: The weight applied to false negatives. 
            - hidden_dim: The number of nodes in the second linear layer of the two-layer classifier.
            - latent_dim: The dimensionality of the input embedding. 
            - dropout
        '''
        # Initialize the torch Module
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid())

        self.init_weights()
        self.to(DEVICE)

        self.loss_func = WeightedBCELoss(weight=bce_loss_weight)

    def init_weights(self):
        '''Initialize model weights according to which activation is used.'''
        torch.nn.init.kaiming_normal_(self.classifier[0].weight)
        torch.nn.init.xavier_normal_(self.classifier[3].weight)

    def reset(self):
        '''Reset the model weights according to the weight initialization function.'''
        self.init_weights()

    def forward(self,
        inputs:torch.FloatTensor,
        batch_size:int=None, **kwargs):
        '''
        A forward pass of the Classifier.'''
        assert inputs.dtype == torch.float32, f'classifiers.Classifier.forward: Expected input embedding of type torch.float32, not {inputs.dtype}.'
        
        if batch_size is None:  # If no batches are specified, treat as a single batch. 
            outputs = self.classifier(inputs) 
        else:
            outputs = [self.classifier(batch) for batch in torch.split(inputs, batch_size)]
            outputs = torch.concat(outputs)

        return outputs

    def get_metrics(self, dataset:Dataset) -> Tuple[float, float, torch.FloatTensor, torch.FloatTensor]:
        '''Calculate the validation loss and accuracy on an input Dataset'''
        self.eval()
        outputs, targets = self(dataset.get_embeddings(), batch_size=32), dataset.get_labels()
        loss, acc = self.loss_func(outputs, targets), get_balanced_accuracy(outputs, targets)
        self.train()
        return loss, acc, outputs, targets

    def test(self,
        test_dataset:torch.utils.data.Dataset,
        threshold:float=0.5) -> TestReporter:
        # reporter:Reporter=None) -> Reporter:
        '''Evaluate the classifier model on the data specified by the DataLoader

        args:
            - dataloader: The DataLoader object containing the training data. 
            - bce_loss_weight: The weight to be passed into the WeightedBCELoss constructor.
            - threshold: The threshold above which (inclusive) a classification is considered a '1.' 
        '''         
        reporter = TestReporter() # Instantiate a Reporter for storing collected data. 

        test_loss, test_acc, outputs, targets = self.get_metrics(test_dataset)
        reporter.add_test_metrics(test_loss, test_acc)
        # Compute the confusion matrix information.
        outputs = outputs.reshape(targets.shape)
        outputs = torch.where(outputs < threshold, 0, 1) 
        # ORDER IS TARGETS, OUTPUTS
        (tn, fp, fn, tp) = sklearn.metrics.confusion_matrix(targets.detach().numpy(), outputs.detach().numpy()).ravel()
        reporter.add_confusion_matrix(tn, fp, fn, tp)

        return reporter

    def fit(self, 
        train_dataloader:torch.utils.data.DataLoader,
        val_dataset:torch.utils.data.Dataset=None,
        epochs:int=300,
        lr:float=0.01) -> TrainReporter:
        '''Train the classifier model on the data specified by the DataLoader.

        args:
            - dataloader: The DataLoader object containing the training data. 
            - val_dataset: The Dataset object containing the validation data. Expected to be a Dataset as defined in datasets.py.
            - epochs: The number of epochs to train for. 
            - lr: The learning rate. 
            - bce_loss_weight: The weight to be passed into the WeightedBCELoss constructor.
        '''
        self.train() # Put the model in train mode.

        reporter = TrainReporter(epochs=epochs, lr=lr, batches_per_epoch=len(list(train_dataloader)))

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        pbar = tqdm(range(epochs), desc='classifiers.Classifier.train_')

        # Want to log the initial training and validation metrics.
        val_loss, val_acc, _, _ = self.get_metrics(val_dataset)
        train_loss, train_acc, _, _ = self.get_metrics(train_dataloader.dataset)
        reporter.add_val_metrics(val_loss, val_acc)
        reporter.add_train_metrics(val_loss, val_acc)
        pbar.set_postfix({'val_acc':np.round(val_acc, 4), 'train_acc':np.round(train_acc, 4)})

        for _ in pbar:

            train_losses, train_accs = [], [] 
            for batch in train_dataloader:
                # Evaluate the model on the batch in the training dataloader. 
                outputs, targets = self(batch['emb']), batch['label']
                train_loss, train_acc = self.loss_func(outputs, targets), get_balanced_accuracy(outputs, targets)
                # Accumulate the results over batches.
                train_losses.append(train_loss.item())
                train_accs.append(train_acc)

                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            reporter.add_train_metrics(np.mean(train_losses), np.mean(train_accs))
            
            # Evaluate the validation metrics and update the progress bar.
            val_loss, val_acc, _, _ = self.get_metrics(val_dataset)
            reporter.add_val_metrics(val_loss, val_acc)
            
            pbar.set_postfix({'val_acc':np.round(val_acc, 4), 'train_acc':np.round(np.mean(train_accs), 4)})
            
        pbar.close()
        
        return reporter

    def save(self, path:str):
        '''Save the model weights to the specified path.'''
        torch.save(self.state_dict(), path)


class SimpleClassifier(Classifier):
    '''Class defining a simplified version of the binary classification head.'''

    def __init__(self, 
        bce_loss_weight:float=1.0,
        latent_dim:int=1):
        '''
        Initializes a single-layer linear classification head. 

        args:
            - bce_loss_weight: The weight applied to false negatives. 
            - latent_dim: The dimensionality of the input embedding. 
        '''
        # Initialize the torch Module. The classifier attribute will be overridden by the rest of this init function. 
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 1),
            torch.nn.Sigmoid())

        self.init_weights()
        self.to(DEVICE)

        self.loss_func = WeightedBCELoss(weight=bce_loss_weight)

    def init_weights(self):
        '''Initialize model weights according to which activation is used.'''
        torch.nn.init.kaiming_normal_(self.classifier[0].weight)

    def reset(self):
        '''Reset the model weights according to the weight initialization function.'''
        self.init_weights()


def optimize_hyperparameters(
        dataloader:torch.utils.data.DataLoader, 
        val:torch.utils.data.DataLoader, 
        model:Classifier=Classifier(), 
        n_calls:int=50,
        verbose:bool=True, 
        epochs:int=5) -> list: # -> skopt.OptimizeResult:

    # Probably keep the weights as integers, at least for now. 
    search_space = [skopt.space.Real(1, 10, name='bce_loss_weight')]

    # NOTE: Can't use val_loss (unless I normalize it or something) because it's always better with loss_weight=1.
    
    # Create the objective function. Decorator allows named arguments to be inferred from search space. 
    @skopt.utils.use_named_args(dimensions=search_space)
    def objective(bce_loss_weight=None):
        # if verbose: print(f'classifiers.optimize_hyperparameters: Testing bce_loss_weight={bce_loss_weight}.')
        model.reset()
        model.fit(dataloader, verbose=False, epochs=epochs, lr=0.01, bce_loss_weight=bce_loss_weight, threshold=0.5)
        
        # Evaluate the performance of the fitted model on the data.
        reporter = model.test(val, bce_loss_weight=bce_loss_weight, threshold=0.5)
        test_acc = reporter.get_test_accs()[0] # Get the test loss following evaluation on the data. 
        # if verbose: print(f'classifiers.optimize_hyperparameters: Recorded a test loss of {np.round(test_loss, 2)} with bce_loss_weight={bce_loss_weight}.')
        if verbose: print(f'\r\r\rclassifiers.optimize_hyperparameters: Recorded a test accuracy of {np.round(test_acc, 2)} with bce_loss_weight={bce_loss_weight}.')
        return -test_acc # Return negative so as not to minimize accuracy/
    
    # We are using loss, so acceptable to just minimize the output of the objective function. 
    result = skopt.gp_minimize(objective, search_space)
    return result.x



# def classifier_run_on_gtdb(gtdb_path):

#     results = {}
#     for genome, row in annotations.iterrows():
#         EmbeddingMatrix = getEmbedding(row.embedding_file)
#         #break
#         labels = classifier(torch.tensor(EmbeddingMatrix.values)).numpy().T[0]
        
#         results = pd.DataFrame({"gene": EmbeddingMatrix.index,  "genome": genome, "selenoprotein": labels})
#         results["gene"] = results['gene'].apply(lambda x: x.split(" ")[0])
#         annots = pd.read_csv(row.annotation_file, sep="\t", skiprows=[1]).dropna()
#         annots["gene name"] = annots["gene name"].apply(lambda x: x.replace(".", "_"))
#         annots = annots.set_index("gene name")
#         hits = results[results.selenoprotein > 0.5]
#         hits_with_annotation = hits.set_index("gene").join(annots)
#         ko_map = hits_with_annotation.dropna().groupby("KO").count()["genome"].to_dict()
        
#         # Record the required details
#         results_dict[genome] = {
#             "total_hits": len(hits),
#             "hits_with_annotation": len(hits_with_annotation.dropna()),
#             "total_genes": EmbeddingMatrix.shape[0],
#             "total_genes_with_annotation": len(annots),
#             "hits_with_annotation": ko_map,
#             "seld_copy_num":len(annots[annots.KO == "K01008"])
#         }


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
 
#         self.add_state('total', torch.Tensor(0).to(DEVICE), dist_reduce_fx='sum')
#         self.add_state('n', torch.Tensor(0).to(DEVICE), dist_reduce_fx='sum')

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
#         weight = torch.where(targets == 1, self.weight, 1).to(DEVICE)
        
#         self.total += torch.sum((ce * weight))
#         self.n += inputs.numel()

 





