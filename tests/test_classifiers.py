'''Tests for the src/classifiers.py file.'''

import unittest

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/prichter/Documents/selenobot/src/')
sys.path.append('/home/prichter/Documents/selenobot/')

import dataset
import utils

# # NOTE: DATA_DIR is specific to a problem (detect, extend, etc.)
# DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/detect'
# filenames = ['train.csv', 'val.csv', 'test.csv']

class TestClassifiers(unittest.TestCase):
    '''A series of test to check that the classifiers are all working OK.'''

    class test_classifiers_aac_tokenizer_length_normalization(self):
        '''Make sure the tokenizer built into the AacClassifier generates the
        expected embeddings.'''

        test_cases = [('AAAAA', 'AA', 'AAA')]
    
    class test_classifiers_aac_tokenizer_integer_conversion(self):
        '''Confirm that the amino acid-to-integer conversion works correctly'''
        pass
