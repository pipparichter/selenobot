'''Plotting utilities.'''

from typing import NoReturn, Tuple, List, Dict

import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn
import re
from tqdm import tqdm
import scipy.stats

from reporter import Reporter

# Some specs to make sure everything is in line with Nature Micro requirements. 
DPI = 500
TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 10


def set_fontsize(ax:plt.Axes) -> NoReturn:
    '''Function for easily adjusting the label font sizes.'''
    ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=LABEL_FONT_SIZE)
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=LABEL_FONT_SIZE)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=LABEL_FONT_SIZE)
    ax.set_title(ax.get_title(), fontsize=TITLE_FONT_SIZE)
    
    return ax


def get_palette(n_colors:int):
    '''Get the color palette from the matplotlib Blues colorset.'''
    return sns.color_palette('Blues', n_colors=n_colors)


def plot_train_curve(reporter:Reporter, path:str=None, title:str='plot.plot_train_curve') -> NoReturn:
    '''Plots information provided in the Reporter object returned by the train_ model method.
    
    args:
        - reporter: The reporter object containing the train information. 
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot.
    '''
    fig, ax = plt.subplots(1)
    
    ax = sns.lineplot(data=reporter.get_loss_info(pool=True), y='value', x='epoch', hue='metric', ax=ax, palette=get_palette(2))
    ax.legend().set_title('') # Turn off legend title because it looks crowded. 

    # Add horizontal lines designating epochs.
    for x in range(reporter.epochs):
        ax.axvline(x=x, ymin=0, ymax=1, color='gray', ls=(0, (1, 5)), lw=1)

    # Make sure all labels are the right size. 
    # ax.set_yscale('log')
    ax.set_title(title)
    ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: str(np.round(x, 2))))
    ax.get_yaxis().set_minor_formatter(mpl.ticker.FuncFormatter(lambda x, pos: str(np.round(x, 2))))
    ax.set_ylabel('loss')
    # plt.ticklabel_format(style='plain', axis='y')

    set_fontsize(ax)

    ax.legend(fontsize=LABEL_FONT_SIZE, bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.5)

    if path is not None:
        fig.savefig(path, format='png', dpi=DPI)



def plot_confusion_matrix(reporter:Reporter, path:str=None, title:str='plot.plot_confusion_matrix', ax:plt.Axes=None) -> NoReturn:
    '''Plots a confusion matrix using a Reporter.
    
    args:
        - reporter: The reporter object containing the train information. 
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot. 
        - ax: A matplotlib axis. If specified, the confusion matrix is added to this axis. 
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

    # Make sure all fonts are correct. 
    set_fontsize(ax)
    
    if path is not None:
        fig.save(path, format='png', dpi=DPI)


def plot_roc_curve_comparison(
        reporters:Tuple[List[Reporter], List[Reporter]],
        labels:Tuple[str, str]=('EmbeddingClassifier', 'AacClassifier'),
        path:str=None, 
        title:str='plot.plot_roc_curve_comparison') -> NoReturn: 
    '''Plot the ROC curve using a Reporter object which contains confusion matrix
    information for a variety of thresholds. 

    thargs:
        - reporters: A tuple of lists of reporters containing test information.
        - labels: The names of the two ROC curves being compared.
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot. 
    '''
    fig, ax = plt.subplots(1)

    # The first set of reporters specified should be the baseline. 
    plot_roc_curve(reporters[0], add_confusion_matrix=False, title='', ax=ax, path=None)
    # Add the inset confusion matrix for the non-baseline ROC curve. 
    plot_roc_curve(reporters[1], add_confusion_matrix=True, title='', ax=ax, path=None)

    # Should be two lines on the plot, and the first should be the baseline. 
    ax.lines[0].set_linestyle('--')
    ax.lines[0].set_color('gray')

    ax.set_title(title)

    # Make sure font sizes are correct. 
    set_fontsize(ax)

    if path is not None:
        fig.savefig(path, format='png', dpi=DPI)


def plot_selenoprotein_ratio_barplot(gtdb_data, title='plot.plot_selenoprotein_ratio_barplot', path=None):
    '''Plots a histogram with selD copy number on the x axis, and the total hits ratio on the y axis.
    This should help visualize how selD copy and selenoprotein content are connected.'''

    fig, ax = plt.subplots(1)
    # fig = plt.figure()

    # Probably want to get the average of the selenoprotein_ratio. 
    plot_data = gtdb_data[['selD_copy_num', 'selenoprotein_ratio']]
    # plot_data = plot_data.groupby('selD_copy_num', as_index=False).mean()
    plot_data = plot_data.groupby('selD_copy_num', as_index=True).selenoprotein_ratio.agg([np.mean, np.std])

    # sns.barplot(data, x='selD_copy_num', y='selenoprotein_ratio', ax=ax, color='cornflowerblue')
    plot_data.plot(kind='bar', y='mean', yerr='std', ax=ax, color='cornflowerblue', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('selenoprotein_ratio')
    ax.set_yticks(np.arange(0, 0.07, 0.01))
    ax.set_yticklabels(np.arange(0, 0.07, 0.01))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right')

    set_fontsize(ax)

    # for container in ax.containers:
    #     ax.bar_label(container, labels=[f'{np.round(100 * x, 2)}%' for x in container.datavalues], fontsize=LABEL_FONT_SIZE)

    ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    ax.legend([])

    if path is not None: # Save the figure if a path is specified.
        fig.savefig(path, format='png')


def plot_selenoprotein_ratio_ecdf(gtdb_data, title='plot.plot_selenoprotein_ratio_ecdf', path=None, add_mannwhitneyu=True):
    '''Plot the ECDF plot of selenoprotein_ratio for each of the selD copy numbers. The code
    assumes that there are only four options for selD copy number.'''

    # assert len(np.unique(gtdb_data.selD_copy_num)) < 6, 'plot.plot_selenoprotein_ratio_ecdf: There are more options for selD_copy_num than expected.'

    fig, ax = plt.subplots()

    # colors = ['cornflowerblue', 'cadetblue', 'royalblue', 'skyblue', 'steelblue']
    n_colors = len(np.unique(gtdb_data.selD_copy_num)) 
    palette = get_palette(n_colors=n_colors)
    # colors = ['skyblue', 'steelblue', 'slategray', 'cornflowerblue', 'royalblue']
    legend = []

    # All MW tests computations are relative to the zero copy case. 
    seld_0 = list(gtdb_data[gtdb_data.selD_copy_num == 0].selenoprotein_ratio)

    for idx, group in gtdb_data.groupby('selD_copy_num'):
        group = group[['selenoprotein_ratio']]
        # Stat is proportion by default. 
        sns.ecdfplot(ax=ax, data=group, x='selenoprotein_ratio', color=palette[idx])
        
        label = str(idx)
        if add_mannwhitneyu: 
            # Is this statistic symmetric? Should make sure this order is correct, perhaps. Ok yes, confirmed that it is symmetric, so I have not messed up. 
            # u = scipy.stats.mannwhitneyu(seld_0, list(group.values))
            # NOTE: Not totally sure why we need a type conversion here. 
            p = scipy.stats.mannwhitneyu(np.ravel(group.values).astype(np.float64), seld_0).pvalue.item()
            # label = label + f' (U={np.round(u.statistic.item(), 2)})'
            if p < 0.001:
                power = int(np.format_float_scientific(p).split('e')[-1])
                label = label + f' (p<1e{power + 1})'
            else:
                label = label + f' (p={np.round(p, 3)})'
        legend.append(label)

    ax.set_ylabel('proportion')
    ax.set_xlim(0, 0.07)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)

    # ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=LABEL_FONT_SIZE)
    # ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=LABEL_FONT_SIZE)

    ax.set_xticklabels(np.arange(0, 0.07, 0.01))
    ax.set_yticklabels(np.round(np.arange(0, 1, 0.1), 2))
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=LABEL_FONT_SIZE)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=LABEL_FONT_SIZE)

    set_fontsize(ax, legend=False)
    ax.legend(legend, title='selD_copy_num', title_fontsize=LABEL_FONT_SIZE, fontsize=LABEL_FONT_SIZE)

    if path is not None:
        fig.savefig(path, format='png')

   
def plot_roc_curve(
        reporters:List[Reporter], 
        path:str=None, 
        title:str='tree.plot_roc_curve', 
        ax:plt.Axes=None,
        add_confusion_matrix:bool=False) -> NoReturn:
    '''Plot the ROC curve using a Reporter object which contains confusion matrix
    information for a variety of thresholds. 

    args:
        - reporters: A list of reporters containing test information.
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot. 
        - ax: A matplotlib axis. If specified, the confusion matrix is added to this axis. 
        - add_confusion_matrix: Whether or not to include the confusion matrix as an inset. 
    '''

    if ax is None:
        fig, ax = plt.subplots(1)

    # First need to organize the information into a DataFrame for plotting. 
    data = {'true_positive_rate':[r.get_true_positive_rate() for r in reporters], 'false_positive_rate':[r.get_false_positive_rate() for r in reporters]}
    # Not totally sure why I am getting NaNs here...
    data = pd.DataFrame(data) # .fillna(0)

    # NOTE: Threshold is an upper bound, so when threshold is 1, everything should be classified as 0. When threshold
    # is zero, everything should be classified as 1.

    sns.lineplot(data=data, y='true_positive_rate', x='false_positive_rate', ax=ax, color='cornflowerblue', legend=None)

    if add_confusion_matrix:
        # Inset axes in the plot showing the confusion matrix. Should be in the bottom right corner. 
        # NOTE: (0,0) is bottom left and (1,1) is top right of the axes. This is the pixel coordinate system of the display. (0,0) is the bottom left and (width, height) is the top right of display in pixels.
        axins = ax.inset_axes([0.5, 0.1, 0.4, 0.5]) #, edgecolor='black')
        plot_confusion_matrix(reporters[3], title='', ax=axins)

    ax.set_title(title)

    # Make sure font sizes are correct. 
    set_fontsize(ax, legend=False)

    if path is not None:
        fig.savefig(path, format='png', dpi=DPI)


# def plot_train_test_val_split(
#     train_data:pd.DataFrame=None, 
#     test_data:pd.DataFrame=None, 
#     val_data:pd.DataFrame=None, 
#     title:str='plot.plot_train_test_val_split',
#     path:str=None) -> None: 
#     '''Plot information about the train-test-validation  split.

#     args:
#         - train_data: A DataFrame containing the train data.   
#         - test_data: A DataFrame containing the test data.     
#         - val_data: A DataFrame containing the validation data.  
#         - title: A title for the plot. 
#         - path: The path to which the file should be written. If None, the figure is not saved. 
#     '''

#     # Things we care about are length distributions, as well as
#     # proportion of negative and positive instances (and selenoproteins, for that matter)

#     fig, ax = plt.subplots(1, figsize=(6, 5)) #, figsize=(15, 10))

#     plot_data = {'dataset':['train', 'test', 'val']}
#     plot_data['truncated'] = [np.sum([1 if '[' in row.Index else 0 for row in data.itertuples()]) for data in [train_data, test_data, val_data]] 
#     plot_data['full_length'] = [len(data) - count for data, count in zip([train_data, test_data, val_data], plot_data['truncated'])]

#     plot_data = pd.DataFrame(plot_data).set_index('dataset')
#     plot_data.plot(kind='bar', ax=ax, color=['cornflowerblue', 'lightsteelblue'], edgecolor='black', linewidth=0.5)

#     sizes = [len(train_data), len(test_data), len(val_data)]
#     for container in ax.containers:
#         ax.bar_label(container, labels=[f'{np.round(100 * x/y, 1)}%' for x, y in zip(container.datavalues, sizes)], fontsize=LABEL_FONT_SIZE)

#     # Make sure all the labels look how I want, and are the correct size. 
#     ax.set_title(title)
#     ax.set_ylabel('count')
#     ax.set_xlabel('') # Remove the x label.

#     ax.legend(fontsize=LABEL_FONT_SIZE)

#     if path is not None:
#         plt.savefig(path, format='png', dpi=DPI)

 