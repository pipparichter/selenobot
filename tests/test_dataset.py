'''Tests for the src/dataset.py file.'''

import unittest
import numpy as np
import pandas as pd
import os
import tqdm
import torch
import dataset

from selenobot.utils import fasta_ids, csv_ids, load_config_paths

UNIPROT_DATA_DIR = load_config_paths()['uniprot_data_dir']
DETECT_DATA_DIR = load_config_paths()['detect_data_dir']
TRAIN_PATH = os.path.join(DETECT_DATA_DIR, 'train.csv')

# Keyword arguments to pass into the get_dataloader function for each test. 
# I just used the length embedder for everything, as these embeddings are the smallest, and all I am testing is Dataloader functionality. 
BATCH_SIZE = 128
GET_DATALOADER_KWARGS = {'verbose':False, 'embedder':'length', 'batch_size':BATCH_SIZE}

# test_[method under test]_[expected behavior]?_when_[preconditions]

# Only ever going to use the DataLoader for the training data, so probably just need to test it on that.

class TestDataset(unittest.TestCase):
    '''A series of tests to ensure that the Dataset object is behaving as expected.'''

    def test_all_proteins_accounted_for_by_dataloader(self):
        '''Make sure all IDs in a file are represented in the DataLoader.'''
        ref_ids = csv_ids(TRAIN_PATH)
        dataloader = dataset.get_dataloader(TRAIN_PATH, balance_batches=False, **GET_DATALOADER_KWARGS)
            
        # Collect all the IDs in the dataloader. 
        ids = [batch['id'] for batch in dataloader]
        ids = np.unique(np.ravel(np.concatenate(ids)))
        # I don't think sampling should be with replacement. 
        assert len(ref_ids) == len(ids), f'Expected {len(ref_ids)} returned by the  DataLoader, but got {len(ids)}.'

    def test_all_proteins_accounted_for_by_balanced_batch_dataloader(self):
        '''Make sure all IDs in a file are represented in the DataLoader.'''
        ref_ids = set(csv_ids(TRAIN_PATH))
        dataloader = dataset.get_dataloader(TRAIN_PATH, balance_batches=True, **GET_DATALOADER_KWARGS)
        # Collect all the IDs in the dataloader. 
        ids = [batch['id'] for batch in dataloader]
        ids = set(np.concatenate(ids).ravel()) 
        num_ids = len(ids) + dataloader.dataset.num_removed # Account for the number of IDs removed for divisible batches. 
        
        assert len(ref_ids) == num_ids, f'Expected {len(ref_ids)} returned by the balanced batch DataLoader, but got {num_ids}.'
        
    def test_balanced_batch_dataloader_yields_correct_selenoprotein_fraction(self):
        '''Make sure the balanced dataloader is returning the correct number of selenoproteins in each batch.'''
        for selenoprotein_fraction in np.arange(0.1, 1, 0.1): # Try for a variety of different fractions, just in case. 
                dataloader = dataset.get_dataloader(TRAIN_PATH, balance_batches=True, selenoprotein_fraction=selenoprotein_fraction, **GET_DATALOADER_KWARGS)
                for batch in dataloader:
                    x = len([id_ for id_ in batch['id'] if '[' in id_]) / BATCH_SIZE
                    assert np.round(x, 1) == np.round(selenoprotein_fraction, 1), f'Expected a selenoprotein fraction of {np.round(selenoprotein_fraction, 1)}, got {np.round(x, 1)}'
    
    def test_balanced_dataloader_samples_evenly_across_selenoproteins(self):
        '''Check that selenoproteins are being re-sampled somewhat evenly (not one protein being sampled many times).'''
        # Don't use sec_ids from sec_truncated.fasta, as we are just looking at the IDs in train.csv.
        sec_ids = [id_ for id_ in csv_ids(TRAIN_PATH) if '[1]' in id_]
        num_sec = len(sec_ids) # Total number of selenoproteins. 

        for selenoprotein_fraction in np.arange(0.1, 0.5, 0.1): # Try for a variety of different fractions, just in case. 
            counts = np.zeros(num_sec).astype(int)
            
            dataloader = dataset.get_dataloader(TRAIN_PATH, balance_batches=True, selenoprotein_fraction=selenoprotein_fraction, **GET_DATALOADER_KWARGS)
            # Get all the gene IDs returned by the DataLoader.
            for batch in dataloader: 
                counts += np.array([list(batch['id']).count(id_) for id_ in sec_ids])
            # Contains the probability of observing each resampling count. 
            expected_resampling = dataloader.dataset.num_resampled / len(sec_ids)
            # tolerance = 0.5 * expected_resampling # Seems reasonable?
            tolerance = 5

            upper_bound, lower_bound = expected_resampling + tolerance, expected_resampling - tolerance

            assert np.all(counts <= upper_bound), f'Resampling of one selenoprotein exceeds upper bound of {np.round(upper_bound, 2)}'
            assert np.all(counts >= lower_bound), f'Resampling of one selenoprotein is below lower bound of {np.round(lower_bound, 2)}'            

if __name__ == '__main__':
    unittest.main()



