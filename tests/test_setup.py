'''Tests to make sure the data setup worked as expected.'''

import unittest

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/prichter/Documents/selenobot/src/')
sys.path.append('/home/prichter/Documents/selenobot/')

import dataset
import utils

# NOTE: DETECT_DATA_DIR is specific to a problem (detect, extend, etc.)
DETECT_DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/detect'
DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/'

filenames = ['train.csv', 'val.csv', 'test.csv']

# test_[method under test]_[expected behavior]?_when_[preconditions]

class TestSetup(unittest.TestCase):

    def test_setup_all_sec_proteins_present_when_data_is_partitioned(self):
        '''Test to make sure all selenoproteins are accounted for in the train, test, and validation data.'''

        # Read in the selenoprotein ids. 
        ref_sec_ids = utils.fasta_ids(os.path.join(DETECT_DATA_DIR, 'sec_trunc.fasta'))

        # Collect the sec IDs from the data files. 
        sec_ids = []
        for filename in filenames:
            data = pd.read_csv(os.path.join(DETECT_DATA_DIR, filename), usecols=['id', 'label'])
            data = data[data['label'] == 1] # Locate the IDs labeled as selenproteins. 
            sec_ids.append(data['id'].values)
        
        sec_ids = np.ravel(np.concatenate(sec_ids))
        
        assert len(sec_ids) == len(np.unique(sec_ids)), 'Some selenoproteins are counted multiple times in the partitioned data.'
        assert len(np.unique(sec_ids)) >= len(ref_sec_ids), f'There are {len(ref_sec_ids) - len(np.unique(sec_ids))} fewer selenoproteins in the datasets than in the original data.'
        assert np.all(np.isin(ref_sec_ids, sec_ids)), 'Not all of the selenoproteins are represented in the partitioned data.'

    def test_setup_label_is_one_when_seq_is_truncated(self):
        '''Test to make sure all selenoproteins are flagged with a 1 in the label column.'''

        for filename in filenames:
            # Can't use the actual sequence, as the U residue has been removed. 
            data = pd.read_csv(os.path.join(DETECT_DATA_DIR, filename), usecols=['id', 'label'])
            for row in data.itertuples():
                if '[' in row.id:
                    assert row.label == 1, f'The selenoprotein {row.id} is mislabeled as full-length.'

    
    def test_setup_label_is_zero_when_seq_is_not_truncated(self):
        '''Test to make sure all full-length proteins are flagged with a 0 in the label column.'''

        for filename in filenames:
            # Can't use the actual sequence, as the U residue has been removed. 
            data = pd.read_csv(os.path.join(DETECT_DATA_DIR, filename), usecols=['id', 'label'])
            for row in data.itertuples():
                if '[' not in row.id:
                    assert row.label == 0, f'The full-length protein {row.id} is mislabeled as truncated.'

    def test_setup_all_sec_proteins_represented_after_truncating(self):
        '''Test to make sure all selenoproteins are present before and after the truncation step.'''

        sec_ids = utils.fasta_ids(os.path.join(DATA_DIR, 'sec.fasta'))
        sec_trunc_ids = utils.fasta_ids(os.path.join(DETECT_DATA_DIR, 'sec_trunc.fasta'))
        # Make sure to remove brackets from all the truncated IDs/ 
        for id_ in sec_trunc_ids:
            assert '[1]' in id_, f'Gene ID {id_} is not marked as truncated, but is present in sec_trunc.csv'
            id_ = id_[:-3] # Remove the [1] at the end of the ID. 
            assert id_ in sec_ids, f'Gene ID {id_} is represented in sec.fasta, but not sec_trunc.fasta.'

    def test_setup_all_proteins_in_all_fasta_represented_after_partitioning(self):

        ref_ids = utils.fasta_ids(os.path.join(DETECT_DATA_DIR, 'all.fasta'))
        ids = [utils.csv_ids(os.path.join(DETECT_DATA_DIR, filename)) for filename in filenames]
        ids = np.ravel(np.concatenate(ids)) # Collect all IDs in a one-dimensional numpy array. 

        assert np.all(np.isin(ref_ids, ids)), 'Proteins present in all.fasta are missing in the partitioned data. '




if __name__ == '__main__':
    unittest.main()

