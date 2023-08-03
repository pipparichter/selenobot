'''
Plotting utilities for the ProTex tool. 
'''
# # Where is tensorflow being imported such that I even need to shut it up??
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.WARN)

import matplotlib.pyplot as plt
import seaborn as sns
import dataset 
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

colors = sns.color_palette('Paired')
palette = 'Paired'




def plot_embeddings(embeddings, n_points=100, labels=None, filename=None, title=None, palette='Paired', **kwargs):
    '''
    Apply PCA dimensionality reduction to the tokenized amino acid sequences (both 
    those generated simply by counting amino acid content, and those generated using ESM). 

    args:
        - embeddings (np.array): An array with dimensions (samples, embedding_dim). 
        - n_points (int): The number of datapoints to plot. 
        - labels (np.array): An array with size len(embeddings). Contains a label for each
            data point. 
        - filename (str): File name under which to save the figure. 
        - title (str): A title for the plot. 
    '''
    # Sample rows from the array without replacement.
    sample_idxs =  np.random.choice(len(embeddings), size=n_points, replace=False)
    embeddings = embeddings[sample_idxs, 1:]
    
    fig, ax = plt.subplots(1)
    ax.set_title(title)
    
    umap = UMAP(n_components=2) # UMAP seems to work a bit better than PCA. 
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    # Convert to a DataFrame to make seaborn plotting easier. 
    data = pd.DataFrame(umap.fit_transform(embeddings), columns=['UMAP 1', 'UMAP 2'])

    if labels is not None:
        data['label'] = np.array(labels)[sample_idxs] # Make sure labels is an array. 
        sns.scatterplot(data=data, x='UMAP 1', y='UMAP 2', ax=ax, hue='label', palette=palette, **kwargs)
    else:
        sns.scatterplot(data=data, x='UMAP 1', y='UMAP 2', ax=ax, palette=palette, **kwargs)

    if filename is not None: # Save the figure, if a filename is specified. 
        fig.savefig(filename, format='png')


def plot_confusion_matrix(preds, labels, filename=None, title=None):
    '''
    Plots the confusion matrix for a set of predictions generated 
    by a specific model. Because, in this case, the problem is one of
    binary classification, this matrix displays true positives, false
    positives, true negatives, and false negatives. 

    args:
        - preds (np.array)
        - labels (np.array)
    '''

    # Confusion matrix whose i-th row and j-th column entry indicates 
    # the number of samples with true label being i-th class and predicted 
    # label being j-th class.
    matrix = confusion_matrix(preds, labels)
    # tn, fp, fn, tp = matrix.ravel()

    fig, ax = plt.subplots(1)
    sns.heatmap(matrix, ax=ax, annot=True)

    ax.set_xlabel('true')
    ax.set_ylabel('predicted')

    # ax.set_xticklabels(['false positive', 'true positive'])
    # ax.set_yticklabels(['true negative', 'false negative'])
    ax.set_title(title)

    if filename is not None:
        fig.savefig(filename, format='png')


def plot_dataset_length_distributions(sec_data, short_data):
    '''
    Visualize the length distributions of the short and truncated selenoproteins. 

    args:
        - sec_data (pd.DataFrame)
        - short_data (pd.DataFrame)
    '''
    plot_data = {}

    # Grab the sequence lengths. 
    sec_lengths = sec_data['seq'].apply(len).to_numpy()
    short_lengths = short_data['seq'].apply(len).to_numpy()
    lengths = np.concatenate([sec_lengths, short_lengths])

    plot_data['length'] = lengths
    plot_data['dataset'] = np.array(['sec'] * len(sec_lengths) + ['short'] * len(short_lengths))
    plot_data = pd.DataFrame(plot_data) # Convert to DataFrame. 

    fig, ax = plt.subplots(1)

    # Add some relevant information to the plot. 
    ax.text(700, 1000, f'sec.fasta size: {len(sec_lengths)}\nshort.fasta size: {len(short_lengths)}', size='small', weight='semibold')
    ax.set_title('Length distributions over sequence datasets')

    sns.histplot(data=plot_data, hue='dataset', x='length', legend=True, ax=ax, palette=palette, multiple='dodge', bins=15)

    fig.savefig('dataset_length_distributions.png', format='png')


def plot_train_test_composition(train_data, test_data, train_labels, test_labels):
    '''
    '''
    # Label is 1 if the sequence is a selenoprotein. 
    train_sec_count = sum(train_labels)
    test_sec_count = sum(test_labels)

    labels = ['sec', 'short']

    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].pie([train_sec_count, len(train_data) - train_sec_count], labels=labels, autopct='%1.1f%%', colors=colors)
    axes[0].set_title('Training set composition')

    axes[1].pie([test_sec_count, len(test_data) - test_sec_count], labels=labels, autopct='%1.1f%%', colors=colors)
    axes[1].set_title('Testing set composition')

    fig.savefig('train_test_composition.png', format='png')
   


