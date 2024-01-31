'''Code used to generate many of the figures for assessing Selenobot's performance.'''
from typing import NoReturn, Tuple, List, Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats
from selenobot.reporter import Reporter

# Some specs to make sure everything is in line with Nature Micro requirements.
TITLE_FONT_SIZE, LABEL_FONT_SIZE = 10, 10
FIGSIZE = (4, 3)
PALETTE = 'Set1'

# Set all matplotlib global parameters.
plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Arial'], 'size':LABEL_FONT_SIZE})
plt.rc('xtick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('ytick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('axes',  **{'titlesize':TITLE_FONT_SIZE, 'labelsize':LABEL_FONT_SIZE})
# plt.rc('text',  **{'usetex':True}) # Enable use of LaTeX

SAVEFIG_OPTIONS = {'format':'png', 'dpi':500, 'bbox_inches':'tight'}


def plot_train_curve(reporter:Reporter, path:str=None, title:str='', logscale:bool=True) -> NoReturn:
    '''Plots a training curve using information stored in a TrainReporter object.
    
    :param reporter: The TrainReporter object containing the information from model training. 
    :param path: The path where the figure will be saved. If None, the figure is not saved. 
    :param title: A title for the plot.
    :param logscale: Whether or not to use log scale on the y-axis.
    '''
    fig, ax = plt.subplots(1, figsize=FIGSIZE)

    data = reporter.get_loss_info()
    data['metric'] = data['metric'].replace({'val_loss':'validation loss', 'train_loss':'training loss'})
    ax = sns.lineplot(data=data, y='value', x='epoch', hue='metric', ax=ax, palette=PALETTE, style='metric')
    ax.legend().set_title('') # Turn off legend title because it looks crowded. 

    # Add horizontal lines designating epochs.
    for x in range(reporter.epochs):
        ax.axvline(x=x, ymin=0, ymax=1, color='gray', ls=(0, (1, 3)), lw=1)

    ax.set_title(title)

    if logscale:
        ax.set_ylabel('log(loss)')
        plt.yscale('log')
    else:
        ax.set_ylabel('loss')

    ax.legend(fontsize=LABEL_FONT_SIZE, bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.5)

    # Save the figure if a path is provided.
    if path is not None: fig.savefig(path, **SAVEFIG_OPTIONS)


def plot_confusion_matrix(reporter:Reporter, path:str=None, title:str='plot.plot_confusion_matrix', ax:plt.Axes=None) -> NoReturn:
    '''Plots a confusion matrix using a Reporter.

    :param reporter: The TestReporter object containing the results of model testing. 
    :param path: The path where the figure will be saved. If None, the figure is not saved. 
    :param title: A title for the plot. 
    :param : A matplotlib axis. If specified, the confusion matrix is added to this axis. 
    '''

    # Extract the confusion matrix from the reporter object. 
    (tn, fp, fn, tp) = reporter.get_confusion_matrix()
    # Convert the heatmap to a pandas DataFrame for the sake of labels. 

    # Confusion matrix function takes y_predicted and y_true as inputs, which is exactly the output of the predict method.
    if ax is None:
        fig, ax = plt.subplots(1)
        cbar = True
        labels = [[f'true negative ({tn})', f'false positive ({fp})'], [f'false negative ({fn})', f'true positive ({tp})']]
    else: # If the matrix is being plot on an inset axis, shorten the labels and remove the colorbar so it doesn't look crowded.
        cbar = False
        labels = [[f'TN ({tn})', f'FP ({fp})'], [f'FN ({fn})', f'TP ({tp})']]

    # (tn, fp, fn, tp)
    annot_kws = {'fontsize':LABEL_FONT_SIZE}
    sns.heatmap([[tn, fp], [fn, tp]], fmt='', annot=labels, annot_kws=annot_kws, ax=ax, cmap=mpl.colormaps['Blues'], cbar=False, linewidths=0.5, linecolor='black')

    ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    # Make the lines around the confusion matrix visible. 
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    
    if path is not None: fig.save(path, **SAVEFIG_OPTIONS)



def plot_precision_recall_curves(
    reporters:Dict[str, List[Reporter]],
    path:str=None,
    title:str='plot.plot_precision_recall_curve') -> NoReturn:
    '''Plot the ROC curve using a Reporter object which contains confusion matrix
    information for a variety of thresholds. 

    args:
        - reporters: A dictionary mapping curve labels (e.g. 'plm') to lists of reporters.
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot. 
        - add_confusion_matrix: Whether or not to include the confusion matrix as an inset. 
    '''
    fig, ax = plt.subplots(1, figsize=FIGSIZE)

    # Long-form label options to make the plot nicer while keeping the notebook clean. 
    label_map = {'plm':'protein language model', 'length':'sequence length', 'aac':'amino acid content'}

    df = {'precision':[], 'recall':[], 'label':[]}
    for l, rs in reporters.items():
        df['precision'] += [r.get_precision() for r in rs]
        df['recall'] += [r.get_recall() for r in rs]
        df['label'] += [label_map[l]] * len(rs)
    df = pd.DataFrame(df)

    sns.lineplot(data=df, y='precision', x='recall', hue='label', style='label', ax=ax, palette=PALETTE, legend='auto')
    ax.legend(title='')

    ax.set_title(title)

    if path is not None: fig.savefig(path, **SAVEFIG_OPTIONS)


def plot_receiver_operator_curves(
    reporters:Dict[str, List[Reporter]],
    path:str=None,
    title:str='plot.plot_receiver_operator_curves') -> NoReturn:
    '''Plot the ROC curve using a Reporter object which contains confusion matrix
    information for a variety of thresholds. 

    args:
        - reporters: A dictionary mapping curve labels (e.g. 'plm') to lists of reporters.
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot. 
        - add_confusion_matrix: Whether or not to include the confusion matrix as an inset. 
    '''
    fig, ax = plt.subplots(1, figsize=FIGSIZE)

    # Long-form label options to make the plot nicer while keeping the notebook clean. 
    label_map = {'plm':'protein language model', 'length':'sequence length', 'aac':'amino acid content'}

    df = {'label':[], 'true positive rate':[], 'false positive rate':[]}
    for l, rs in reporters.items():
        df['true positive rate'] += [r.get_true_positive_rate() for r in rs]
        df['false positive rate'] += [r.get_false_positive_rate() for r in rs]
        df['label'] += [label_map[l]] * len(rs)
    df = pd.DataFrame(df)
    
    sns.lineplot(data=df, y='true positive rate', x='false positive rate', hue='label', style='label', ax=ax, palette=PALETTE, legend='auto')
    ax.legend(title='')

    ax.set_title(title)

    if path is not None: fig.savefig(path, **SAVEFIG_OPTIONS)

