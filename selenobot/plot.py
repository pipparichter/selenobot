'''Code used to generate many of the figures for assessing Selenobot's performance.'''
from typing import NoReturn, Tuple, List, Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats
from classifiers import TestReporter, TrainReporter
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Some specs to make sure everything is in line with Nature Micro requirements.
TITLE_FONT_SIZE, LABEL_FONT_SIZE = 10, 10
FIGSIZE = (4, 3)
PALETTE = 'Set1'
# Set all matplotlib global parameters.
plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Arial'], 'size':LABEL_FONT_SIZE})
plt.rc('xtick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('ytick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('axes',  **{'titlesize':TITLE_FONT_SIZE, 'labelsize':LABEL_FONT_SIZE})


def plot_training_curve(reporter:TrainReporter, logscale:bool=True) -> plt.Axes:
    '''Plots a training curve using information stored in a TrainReporter object.
    
    :param reporter: The TrainReporter object containing the information from model training. 
    :param logscale: Whether or not to use log scale on the y-axis.
    '''
    fig, ax = plt.subplots(1, figsize=FIGSIZE)
    ax = sns.lineplot(data=reporter.get_training_curve_data(), y='value', x='epoch', hue='metric', ax=ax, palette=PALETTE, style='metric')
    ax.legend().set_title('') # Turn off legend title because it looks crowded. 

    # Add horizontal lines designating epochs.
    for x in range(reporter.epochs):
        ax.axvline(x=x, ymin=0, ymax=1, color='gray', ls=(0, (1, 3)), lw=1)

    if logscale:
        ax.set_ylabel('log(loss)')
        plt.yscale('log')
    else:
        ax.set_ylabel('loss')

    ax.legend(fontsize=LABEL_FONT_SIZE, bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.5)
    return ax

def plot_calibration_curve(reporter:TestReporter, ax:plt.Axes=None, n_bins:int=10) -> plt.Axes:
    '''Plots a reliability curve using data in the Reporter object.

    :param reporter: The TestReporter object containing the information from model testing. 
    :param ax: If specified, the matplotlib Axes to add the plot to. 
    :param nbins: The number of bins to use when generating data for the calibration curve.
    '''
    if ax is None: # Create a new set of Axes if none are specified. 
        fig, ax = plt.subplots(ax, figsize=FIGSIZE) 

    # Reporter stores Sigmoid outputs from the classifier (threshold not applied).
    prob_true, prob_pred = calibration_curve(reporter.targets, reporter.outputs, n_bins=n_bins, strategy='quantile')
    
    # prob_true is the proportion of samples whose class is the positive class in each bin (the y-axis).
    # prob_pred is the mean predicted probability in each bin (the x-axis).

    ax.plot(prob_pred, prob_true)
    ax.set_xlabel('mean predicted probability')
    ax.set_ylabel('fraction of positives')


def plot_confusion_matrix(reporter:TestReporter, threshold:float=0.5) -> plt.Axes:
    '''Plots a confusion matrix using a TestReporter.

    :param reporter: The TestReporter object containing the results of model testing. 
    :param threshold: The threshold to apply to the model output. Any output greater than or
        equal to this threshold value is considered to indicate a truncated selenoprotein.
    '''
    # Apply the threshold to the output values. 
    outputs = np.ones(reporter.outputs.shape)
    outputs[np.where(reporter.outputs < threshold)] = 0

    # Calculate the confusion matrix using the stored output and target values.
    # Make sure to pass targets, outputs to the confusion matrix function in the correct order.
    (tn, fp, fn, tp) = confusion_matrix(reporter.targets, outputs).ravel()
 
    # Confusion matrix function takes y_predicted and y_true as inputs, which is exactly the output of the predict method.
    fig, ax = plt.subplots(1)
    cbar = True
    labels = [[f'true negative ({tn})', f'false positive ({fp})'], [f'false negative ({fn})', f'true positive ({tp})']]

    # (tn, fp, fn, tp)
    annot_kws = {'fontsize':LABEL_FONT_SIZE}
    sns.heatmap([[tn, fp], [fn, tp]], fmt='', annot=labels, annot_kws=annot_kws, ax=ax, cmap=mpl.colormaps['Blues'], cbar=True, linewidths=0.5, linecolor='black')
    ax.set_xticks([])
    ax.set_yticks([])

    # Make the lines around the confusion matrix visible. 
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    
    return ax


def plot_precision_recall_curve(reporter:TestReporter, ax:plt.Axes=None) -> NoReturn:
    '''Plot a precision recall curve using using a TestReporter object. 
    
    :param reporter: The TestReporter object containing the results of model testing. 
    :param ax: If specified, the matplotlib Axes to add the plot to.  
    '''
    if ax is None: # Create a new set of axes if none are provided. 
        fig, ax = plt.subplots(1, figsize=FIGSIZE)

    precision, recall, _ = precision_recall_curve(reporter.targets, reporter.outputs)
    # df = {'precision':precision, 'recall':recall}
    # df = pd.DataFrame(df)
    # sns.lineplot(data=df, y='precision', x='recall', ax=ax, palette=PALETTE)
    ax.plot(recall, precision)
    ax.set_ylabel('precision')
    ax.set_xlabel('recall')


def plot_receiver_operator_curve(reporter:TestReporter, ax:plt.Axes=None) -> NoReturn:
    '''Plot a ROC curve using a TestReporter object. 

    :param reporter: The TestReporter object containing the results of model testing. 
    :param ax: If specified, the matplotlib Axes to add the plot to.  
    ''' 
    if ax is None:
        fig, ax = plt.subplots(1, figsize=FIGSIZE)

    fpr, tpr, _ = roc_curve(reporter.targets, reporter.outputs, pos_label=1) 
    # df = {'true positive rate':tpr, 'false positive rate':fpr}
    # df = pd.DataFrame(df)
    # sns.lineplot(data=df, y='true positive rate', x='false positive rate', ax=ax, palette=PALETTE)
    ax.plot(fpr, tpr)
    ax.set_ylabel('true positive rate')
    ax.set_xlabel('false positive rate')


