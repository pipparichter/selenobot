'''
Plotting utilities for the ProTex tool. 
'''
import sys
sys.path.append('/home/prichter/Documents/selenobot/src/')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.metrics import confusion_matrix
import sklearn.decomposition


palette = 'Set2'


def plot_distribution(data, ax=None, path=None, title=None, xlabel=None, **kwargs):
    '''Visualize the distribution of a set of data. 


    '''
    if ax is None:
        fig, ax = plt.subplots(1)

    sns.histplot(data=data, ax=ax, legend=False,  **kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)

    # If the data passed into the function is more than one-dimensional, 
    # apply dimensionality reduction to one dimension. 
    if len(data.shape) > 1 and (data.shape[-1] > 1):
        pca = sklearn.decomposition.PCA(n_components=1)
        data = pca.fit_transform(data)

    # Data should be one-dimensional following PCA reduction. 
    data = np.ravel(data)

    if path is not None:   
        try:
            fig.savefig(path, format='png')
        except NameError:
            print('Figure could not be saved.')


def plot_distributions(data, title=None, path=None, xlabel=None, **kwargs):
    '''Plots multiple distributions on the same histogram by calling the
    plot_distribution function on each dataset specified.
    
    args:
        - data (tuple): A tuple where each element is a dataset whose
            distribution to plot. 
    ''' 
    assert type(data) == tuple
    assert len(data) > 1 # Make sure more than one dataset is speciied.
    assert np.all([d.shape[-1] == data[0].shape[-1] for d in data])

    # We will want to reduce each of the datasets in the same PCA pass. 
    
    # Get the sizes of each dataset passed in to the function. 
    # This will allow us to extract individual datasets after combination. 
    sizes = [len(d) for d in data]
    data = np.concatenate(data, axis=0)
    # If data is multi-dimensional, reduce using PCA. 
    if len(data.shape) > 1 and (data.shape[-1] > 1):
        pca = sklearn.decomposition.PCA(n_components=1)
        data = pca.fit_transform(data)

    # Data should be one-dimensional following PCA reduction. 
    data = np.ravel(data)

    fig, ax = plt.subplots(1)

    idx = 0
    for size in sizes:
        plot_distribution(data[idx:idx + size], ax=ax, **kwargs)
        idx += size

    ax.set_title(title)
    ax.set_xlabel(xlabel)

    if path is not None:
        fig.savefig(path, format='png')

    


def plot_assess_filter_sprot(data):
    ''''''''
    pass


def plot_filter_sprot(data, idxs=None, clusters=None, path=None):
    '''Visualizes the filtering procedure carried out by the filter_sprot
    function. Plots both the K-means clusters and the data which "passes"
    the filter.

    args:
        - data (np.array): An array containing the SwissProt embeddings before 
            being filtered.
        - idxs (np.array): The filter indices. 
        - clusters (np.array): The cluster label for each sequence embedding in data.
        - path (str): The path for where to save the generated figure. 
    '''

    assert idxs is not None
    assert clusters is not None
    
    idxs = np.concatenate([np.arange(len(data)), idxs])
    n, m = len(data), len(idxs) # n is the number of non-filtered elements. 

    # Use PCA to put the data in two-dimensional space. 
    pca = sklearn.decomposition.PCA(n_components=2)
    data = pca.fit_transform(data)

    print('PCA decomposition of SwissProt data completed.')

    df = pd.DataFrame({'PCA 1':data[idxs, 0], 'PCA 2':data[idxs, 1], 'cluster':clusters[idxs], 'filter':[0]*n + [1]*(m - n)})

    fig, axes = plt.subplots(2, figsize=(8, 10))
    # sns.scatterplot(data=df x='UMAP 1', y='UMAP 2', ax=ax, hue='label', legend=False, **kwargs)
    sns.scatterplot(data=df, x='PCA 1', y='PCA 2', ax=axes[0], hue='filter', legend=False, s=2, palette={0:'gray', 1:'black'})
    sns.scatterplot(data=df, x='PCA 1', y='PCA 2', ax=axes[1], hue='cluster', legend=False, s=2, palette='Set2')
    axes[0].set_title('Selecting representative sequences from SwissProt')

    fig.savefig(path, format='png')


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
   


