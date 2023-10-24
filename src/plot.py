'''Plotting utilities.'''

from typing import NoReturn, Tuple, List, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats

from reporter import Reporter

# Some specs to make sure everything is in line with Nature Micro requirements.
DPI = 500
TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 10
FIGSIZE = (4, 3)
FONT = 'Arial'
PALETTE = 'Set1'

# Set all matplotlib global parameters.
plt.rc('font', **{'family':'sans-serif', 'sans-serif':[FONT], 'size':LABEL_FONT_SIZE})
plt.rc('xtick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('ytick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('axes',  **{'titlesize':TITLE_FONT_SIZE, 'labelsize':LABEL_FONT_SIZE})

savefig_options = {'format':'png', 'dpi':DPI, 'bbox_inches':'tight'}


def plot_train_curve(reporter:Reporter, path:str=None, title:str='plot.plot_train_curve') -> NoReturn:
    '''Plots information provided in the Reporter object returned by the train_ model method.
    
    args:
        - reporter: The reporter object containing the train information. 
        - path: The path to which the file should be written. If None, the figure is not saved. 
        - title: A title for the plot.
    '''
    fig, ax = plt.subplots(1, figsize=FIGSIZE)

    data = reporter.get_loss_info(pool=True)
    data['metric'] = data['metric'].replace({'val_losses':'validation loss', 'train_losses':'training loss'})
    ax = sns.lineplot(data=data, y='value', x='epoch', hue='metric', ax=ax, palette=PALETTE, style='metric')
    ax.legend().set_title('') # Turn off legend title because it looks crowded. 

    # Add horizontal lines designating epochs.
    for x in range(reporter.epochs):
        ax.axvline(x=x, ymin=0, ymax=1, color='gray', ls=(0, (1, 3)), lw=1)

    # Make sure all labels are the right size. 
    # ax.set_yscale('log')
    ax.set_title(title)

    def tick_format(x, pos):
        '''Function for formatting the tick labels.'''
        tick = str(np.round(x, 2))
        # Only display the tick mark if there is one decimal place of precision after rounding.
        if len(tick.split('.')[-1]) > 1:
            tick = '' 
        return tick

    ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(tick_format))
    ax.get_yaxis().set_minor_formatter(mpl.ticker.FuncFormatter(tick_format))
    ax.set_ylabel('loss')
    # plt.ticklabel_format(style='plain', axis='y')

    ax.legend(fontsize=LABEL_FONT_SIZE, bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.5)

    if path is not None: fig.savefig(path, **savefig_options)


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
    
    if path is not None: fig.save(path, **savefig_options)


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

    if path is not None: fig.savefig(path, **savefig_options)


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

    ax.legend(legend, title='selD_copy_num', title_fontsize=LABEL_FONT_SIZE, fontsize=LABEL_FONT_SIZE)

    if path is not None: fig.savefif(path, **savefig_options)



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

    if path is not None: fig.savefig(path, **savefig_options)


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

    if path is not None: fig.savefig(path, **savefig_options)
