'''Defining a class used to more easily manage data reporting on the performance of a Classifier.'''
import numpy as np
import pandas as pd
from typing import List, NoReturn, Tuple, Dict


class Reporter():
    '''A class for storing information associated with model performance.'''

    def __init__(self):

        # Metrics will be added as methods are called.
        self.loss_metrics = {}
        self.acc_metrics = {}

    def add(self, value:float=None, metric:str=None, group:dict=None):
        '''Add a value under the specified metric in the internal list given by 
        the 'group' keyword argument.
        
        args:
            - value: The loss or value accuracy to add to the instance. 
            - metric: The name of the metric which the value belongs to. 
            - group: The group (either self.loss_metrics or self.acc_metrics) which the metric belongs to. 
        '''
        if metric not in group:
            group[metric] = []
        group[metric].append(value)


class TrainReporter(Reporter):
    '''A class for managing the results of training a Classifier.'''

    def __init__(self,
        epochs:int=None,
        lr:float=None,
        batches_per_epoch:int=None):
        '''Initialize a TrainReporter object.'''

        super().__init__()

        self.epochs = epochs
        self.lr = lr
        self.batches_per_epoch = batches_per_epoch

    def add_train_loss(self, loss) -> NoReturn:
        '''Add a train_loss to the internal list.'''
        self.add(value=loss.item(), metric='train_loss', group=self.loss_metrics)

    def add_train_acc(self, acc) -> NoReturn:
        '''Add a train_acc to the internal list.'''
        self.add(value=acc, metric='train_acc', group=self.acc_metrics)

    def add_train_metrics(self, loss, acc) -> NoReturn:
        '''Add a train_acc and train_loss to the internal lists.'''
        self.add_train_loss(loss)
        self.add_train_acc(acc)

    def add_val_loss(self, loss) -> NoReturn:
        '''Add a val_loss to the internal list.'''
        self.add(value=loss.item(), metric='val_loss', group=self.loss_metrics)

    def add_val_acc(self, acc) -> NoReturn:
        '''Add a val_acc to the internal list.'''
        self.add(value=acc, metric='val_acc', group=self.acc_metrics)

    def add_val_metrics(self, loss, acc) -> NoReturn:
        '''Add a val_acc and val_loss to the internal lists.'''
        self.add_val_loss(loss)
        self.add_val_acc(acc)

    # def pool(self):
    #     '''Pool the data stored as train_loss and train_acc over epochs.'''
    #     train_losses, train_accs =self.loss_metrics['train_loss'], self.acc_metrics['train_acc'] 
    #     self.loss_metrics['train_loss_pooled'] = [np.mean(train_losses[i:i + self.batches_per_epoch]) for i in range(0, len(train_losses), self.batches_per_epoch)]
    #     self.acc_metrics['train_acc_pooled'] = [np.mean(train_accs[i:i + self.batches_per_epoch]) for i in range(0, len(train_accs), self.batches_per_epoch)]

    def _get_info(self, metrics:Dict[str, List[float]]) -> pd.DataFrame:
        '''Use the information returned by the train function to construct a DataFrame for plotting loss. Pools
        the training loss over epochs.'''
        metrics['epoch'] = list(range(self.epochs + 1))
        df = pd.DataFrame(metrics)
        df = df.melt(id_vars=['epoch'], value_vars=df.columns, var_name='metric', value_name='value')

        return df

    def get_loss_info(self) -> pd.DataFrame:
        '''Return a DataFrame containing loss information for plotting a training curve.'''
        return self._get_info(metrics=self.loss_metrics)

    def get_acc_info(self) -> pd.DataFrame:
        '''Return a DataFrame containing loss information for plotting a training curve.'''
        return self._get_info(metrics=self.acc_metrics)


class TestReporter(Reporter):
    '''A class for managing the results of evaluating a Classifier on test data.'''

    def __init__(self):
        '''Initialize a TestReporter.'''
        
        super().__init__()

        self.confusion_matrix = None

    def add_confusion_matrix(self, tn:int, fp:int, fn:int, tp:int) -> NoReturn:
        '''Add new confusion matrix data to the internal list.'''
        self.confusion_matrix = (tn, fp, fn, tp)

    def get_confusion_matrix(self) -> tuple:
        '''Return confusion matrix stored as an attribute.'''
        return self.confusion_matrix

    def add_test_loss(self, loss) -> NoReturn:
        '''Add a train_loss to the internal list.'''
        self.add(value=loss.item(), metric='test_losses', group=self.loss_metrics)

    def add_test_acc(self, acc) -> NoReturn:
        '''Add a train_acc to the internal list.'''
        self.add(value=acc, metric='test_accs', group=self.acc_metrics)

    def add_test_metrics(self, loss, acc) -> NoReturn:
        '''Add a test_acc and test_loss to the internal lists.'''
        self.add_test_loss(loss)
        self.add_test_acc(acc)

    def get_test_loss(self) -> float:
        '''Return the test loss list from the reporter object.'''
        return self.loss_metrics['test_loss'][0] # Should only be one element in this list. 

    def get_test_accs(self) -> float:
        '''Return the test accuracy list from the reporter object.'''
        return self.acc_metrics['test_acc'][0] # Should only have one element in this list.

    def get_true_positive_rate(self) -> float:
        '''Calculate the true positive rate.'''
        _, _, fn, tp = self.get_confusion_matrix()
        tpr = tp / (tp + fn)
        return tpr

    def get_precision(self) -> float:
        '''Calculate the precision.'''
        _, fp, _, tp = self.get_confusion_matrix()
        if fp + tp == 0:
            precision = 1
        else:
            precision = tp / (fp + tp)
        return precision

    def get_recall(self) -> float:
        '''Calculate the recall.'''
        _, _, fn, tp = self.get_confusion_matrix()
        if tp + fn == 0:
            recall = 1
        else:
            recall = tp / (tp + fn)
        return recall
 
    def get_false_positive_rate(self) -> float:
        '''Calculate the false positive rate.'''
        tn, fp, _, _ = self.get_confusion_matrix()
        fpr = fp / (fp + tn)
        return fpr
 
