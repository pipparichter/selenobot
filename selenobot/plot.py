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










