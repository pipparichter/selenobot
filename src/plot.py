import sys
sys.path.append('/home/prichter/Documents/selenobot/src/')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
import matplotlib as mpl
import sklearn
import reporter

import utils
from tqdm import tqdm

cmap = mpl.colormaps['Pastel2']
palette = 'Set2'

# TODO: Might be worth making an info object for plotting results of training. 


# TODO: Thinking about switching this over to plot from file paths. 
def plot_train_test_val_split(
    train_data:pd.DataFrame=None, 
    test_data:pd.DataFrame=None, 
    val_data:pd.DataFrame=None, 
    path:str=None) -> None: 
    '''Plot information about the train-test-validation  split.

    args:
        - train_data: A DataFrame containing the train data.   
        - test_data: A DataFrame containing the test data.     
        - val_data: A DataFrame containing the validation data.  
        - path: The path to which the file should be written. If None, the figure is not saved. 
    '''

    # Things we care about are length distributions, as well as
    # proportion of negative and positive instances (and selenoproteins, for that matter)

    plt.figure(figsize=(15, 10))
    # Establish the subplots. First and second axes are pie charts, the third are length distributions. 
    axes = [plt.subplot(2, 3, 1), plt.subplot(2, 3, 2),plt.subplot(2, 3, 3), plt.subplot(2, 3, (4, 6))]
    titles = ['Training set composition', 'Testing set composition', 'Validation set composition']
    
    # Make the composition pie charts first. 
    for ax, data, title in zip(axes[:3], [train_data, test_data, val_data], titles):
        sec_count = np.sum([1 if '[' in row.Index else 0 for row in data.itertuples()])
        ax.pie([sec_count, len(data) - sec_count], labels=[f'truncated ({sec_count})', f'full-length ({len(data) - sec_count})'], autopct='%1.1f%%', colors=cmap(np.arange(3)))
        ax.set_title(title)

    data = {}
    data['train'] = np.array([len(s) for s in train_data['seq']])
    data['test'] = np.array([len(s) for s in test_data['seq']])
    data['val'] = np.array([len(s) for s in val_data['seq']])

    sns.histplot(data=data, ax=axes[-1], legend=True, stat='count', multiple='dodge', bins=50, palette='Pastel2', ec=None)
    axes[-1].set_yscale('log')
    axes[-1].set_xlabel('lengths')
    axes[-1].set_ylabel('log(count)')
    axes[-1].set_title('Length distributions')

    # Fix the layout and save the figure in the buffer.
    plt.tight_layout()
    
    if path is not None:
        plt.savefig(path, format='png')


def plot_model_training(reporter:reporter.Reporter, path=None): # include=['train_loss', 'pooled_train_loss', 'val_loss']):
    '''Plots information provided in the Reporter object returned by the train_ model method.'''

    # fig, axes = plt.subplots(2, figsize=(12, 10), sharex=True)
    fig, axes = plt.subplots(2, figsize=(20, 10), sharex=True)

    # Seems as though model performance (as inficated by train loss) may vary substantially
    # between batches with and without a selenoprotein present. Might be helpful to label with this in mind. 

    # Pool the train accuracy and losses
    # info = pool_train_info(info)

    loss_df = reporter.get_loss_info()
    acc_df = reporter.get_acc_info()

    sns.lineplot(data=loss_df, y='value', x='batch', hue='metric', ax=axes[0], palette=palette)
    sns.lineplot(data=acc_df, y='value', x='batch', hue='metric', ax=axes[1], palette=palette)

    axes[0].set_title(f"Weighted BCE loss (weight={reporter.bce_loss_weight})")
    axes[0].set_yscale('log')
    axes[0].set_ylabel('log(loss)')
    # axes[0].set_xlabel('batch')

    axes[1].set_title('Accuracy')
    axes[1].set_ylabel('accuracy')

    for ax in axes:
        ax.vlines(reporter.get_epoch_batches(), *ax.get_ylim(), linestyles='dotted', color='LightGray')
        ax.legend().set_title('')

    if path is not None:
        fig.savefig(path, format='png')



def plot_confusion_matrix(reporter:reporter.Reporter, path:str=None):
    ''''''
    # Confusion matrix function takes y_predicted and y_true as inputs, which is exactly the output of the predict method.
    pass




def plot_homology_clusters(clstr_path:str=None, fasta_path:str=None, path:str=None) -> None:
    '''Plot information relating to CD-HIT performance on the data. 
    
    args:
        - clstr_path: The path to the clstr file produced by the program. 
        - fasta_path: The path to the FASTA file from which the program generated clusters. 
        - c: Percent similarity, one of the inputs to CD-HIT. 
        - l: Minimum sequence length, one of the inputs to CD-HIT. 
        - n: Word length, one of the inputs to CD-HIT. 
        - path: The path to which the file should be written. If None, the figure is not saved. 
    '''
    plt.figure(figsize=(18, 12))
    # Set a title for the entire figure. 
    # plt.suptitle('Homology cluster information')
    
    # Establish the subplots. plot on the first row is a cluster size distribution, and the two plots below are scatterplots.
    axes = [plt.subplot(2, 2, (1, 2)), plt.subplot(2, 2, 3), plt.subplot(2, 2, 4)]

    plot_data = {'label':[], 'sec_content':[], 'mean_length':[], 'cluster_size':[]}

    clstr_data = utils.pd_from_clstr(clstr_path) # Read in the clstr information.
    # clstr_data = utils.pd_from_clstr(clstr_path)[:100] # Read in the clstr information.
    fasta_data = utils.pd_from_fasta(fasta_path)

    for label, data in tqdm(clstr_data.groupby('cluster'), desc='plot.plot_homology_clusters'):

        # Worried that grabbing the sequences from FASTA might be excessively slow.  
        idxs = np.where(np.isin(fasta_data.index, data.index))[0]
        seqs = fasta_data.iloc[idxs]['seq'].values

        size = len(data) # Get the length of the cluster. 
        sec_content = len([id_ for id_ in data.index if '[' in id_]) / size

        mean_length = np.mean([len(seq) for seq in seqs])


        plot_data['label'].append(label)
        plot_data['cluster_size'].append(size)
        plot_data['sec_content'].append(np.round(sec_content, 1))
        plot_data['mean_length'].append(mean_length)

    # First deal with plots on the first row. 

    # Plot on the top distribution of cluster size. 
    sns.histplot(data=plot_data['cluster_size'], ax=axes[0], legend=False, bins=100, color='seagreen')#, c='seagreen')
    axes[0].set_xlabel('cluster_size')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('log(count)')
    axes[0].set_title('Cluster size distribution')

    # Minimum cluster size for displaying on the scatterplots. 
    min_cluster_size = 2
    # min_cluster_size_filter = 

    # Additional plots to show how selenoprotein content and sequence length are represented over clusters. 
    plot_data = pd.DataFrame(plot_data)

    # Want to bin the data to make hues look nicer. 
    min_mean_length, max_mean_length = min(plot_data['mean_length']), max(plot_data['mean_length'])
    # mean_length_step = (max_mean_length - min_mean_length) // 10
    # bins =  np.concatenate([np.arange(0, max_mean_length, 250), np.array([max_mean_length])])
    bins =  np.arange(0, max_mean_length, 500)
    bin_labels = ['< 500'] + [f'{int(bins[i])} < length < {int(bins[i + 1])}' for i in range(1, len(bins) - 2)] + [f' > {int(bins[-1])}'] 
    plot_data['mean_length'] = pd.cut(plot_data['mean_length'], bins=bins) # , labels=bin_labels)

    # cmap = sns.light_palette("seagreen", as_cmap=True).resampled(len(bin_labels))# (np.arange(len(bin_labels)))

    sns.scatterplot(data=plot_data, x='label', y='cluster_size', hue='mean_length', ax=axes[1], s=6, edgecolor=None, palette=sns.light_palette('seagreen', n_colors=len(bin_labels)), legend='auto')
    axes[1].set_title('Mean sequence length across clusters')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('log(cluster_size)')
    axes[1].legend(labels=bin_labels)

    sns.scatterplot(data=plot_data, x='label', y='cluster_size', hue='sec_content', ax=axes[2], s=6, edgecolor=None, palette=sns.light_palette('seagreen', n_colors=11), legend='auto')
    axes[2].set_title('Selenoprotein content across clusters')
    axes[2].set_yscale('log')
    axes[2].set_ylabel('log(cluster_size)')

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format='png')



def plot_roc_curve():
    pass