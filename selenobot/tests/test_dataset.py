'''Tests for the src/dataset.py file.'''

import unittest

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/home/prichter/Documents/selenobot/src/')
sys.path.append('/home/prichter/Documents/selenobot/')

import dataset
import utils

import torch

# NOTE: DATA_DIR is specific to a problem (detect, extend, etc.)
DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/detect'
filenames = ['train.csv', 'val.csv', 'test.csv']

# test_[method under test]_[expected behavior]?_when_[preconditions]

class TestDataset(unittest.TestCase):
    '''A series of tests to ensure that the Dataset object is behaving as expected.'''

    def test_dataset_all_ids_present_when_dataloader_created(self):
        '''Make sure all IDs in a file are represented in the DataLoader.'''

        for filename in filenames:
            ref_ids = utils.csv_ids(os.path.join(DATA_DIR, filename))
            dataloader = dataset.get_dataloader(os.path.join(DATA_DIR, filename))
            
            # Collect all the IDs in the dataloader. 
            ids = []
            for batch in dataloader:
                ids.append(batch['id']) # Should be returing numpy arrays. 
            ids = np.ravel(np.concatenate(ids))
            # I don't think sampling should be with replacement. 
            assert len(np.unique(ids)) == len(ids), 'The same ID is being returned multiple times by the DataLoader.'
            assert len(np.unique(ids)) == len(ref_ids), f'The IDs returned from the DataLoader do not match those stored in {filename}.'

    def test_dataset_labels_correct_when_dataloader_created(self):
        '''Make sure labels returned by the DataLoader match those in the file.'''

        for filename in filenames:

            ref_ids = utils.csv_ids(os.path.join(DATA_DIR, filename))
            ref_labels = utils.csv_labels(os.path.join(DATA_DIR, filename))
            id_to_label_map = {ref_ids[i]:ref_labels[i] for i in range(len(ref_ids))}

            dataloader = dataset.get_dataloader(os.path.join(DATA_DIR, filename))
            
            # Collect all the IDs in the dataloader. 
            for batch in dataloader:
                for id_, label in zip(batch['id'], batch['label']):
                    assert id_to_label_map[id_] == label, f'Label returned by DataLoader does not match label for {id_} in {filename}.'
                    # Also double-check to make sure that the gene IDs marked as truncated are labeled correctly.
                    if '[' in id_:
                        assert label == 1, f'Label returned by DataLoader does not match expected label 1 based on the gene ID {id_} in {filename}.'
                    else:
                        assert label == 0, f'Label returned by DataLoader does not match expected label 1 based on the gene ID {id_} in {filename}.'
    
    def test_dataset_all_indices_covered_by_balanced_batch_sampler(self):

        for filename in filenames:
            n = utils.csv_size(os.path.join(DATA_DIR, filename)) # Size of the original dataset. 

            # Balance set to true ensures that the custom dataloader is used. 
            dataloader = dataset.get_dataloader(os.path.join(DATA_DIR, filename), balance=True)
            idxs = [batch['idx'] for batch in dataloader] # Should be a list of tensors. 
            idxs = torch.unique(torch.cat(idxs, axis=0))

            assert len(idxs) == n, f'{len(idxs)} unique indices returned by BalancedBatchSampler do not cover the entire dataset of size {n}.'


if __name__ == '__main__':
    unittest.main()



