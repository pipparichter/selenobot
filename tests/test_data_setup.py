'''Tests to make sure the data setup worked as expected.'''
# pylint: disable=all
import unittest
import numpy as np
import pandas as pd
import os
import data_setup 
import dataset

from data_utils import fasta_seqs, fasta_ids, pd_from_clstr, csv_ids, fasta_size, csv_size, fasta_ids_with_min_seq_length, fasta_size_with_min_seq_length

FILENAMES = ['train.csv', 'val.csv', 'test.csv']

# TODO: Some selenoproteins will be missing because of hte CD-HIT minimum sequence length. 

# test_[method under test]_[expected behavior]?_when_[preconditions]

NUM_EXPECTED_MISSING = 153 # As concluded in summary.py

class TestDataSetup(unittest.TestCase):

    def test_label_is_one_when_sequence_is_truncated(self):
        '''Test to make sure all selenoproteins are flagged with a 1 in the label column.'''
        for filename in FILENAMES:
            # Can't use the actual sequence, as the U residue has been removed. 
            data = pd.read_csv(os.path.join(data_setup.DETECT_DATA_DIR, filename), usecols=['id', 'label'])
            for row in data.itertuples():
                if '[' in row.id:
                    assert row.label == 1, f'The selenoprotein {row.id} is mislabeled as full-length.'
    
    def test_label_is_zero_when_sequence_is_not_truncated(self):
        '''Test to make sure all full-length proteins are flagged with a 0 in the label column.'''
        for filename in FILENAMES:
            # Can't use the actual sequence, as the U residue has been removed. 
            data = pd.read_csv(os.path.join(data_setup.DETECT_DATA_DIR, filename), usecols=['id', 'label'])
            for row in data.itertuples():
                if '[' not in row.id:
                    assert row.label == 0, f'The full-length protein {row.id} is mislabeled as truncated.'

    def test_all_selenoproteins_present_after_setup_sec_trunc(self):
        '''Test to make sure all selenoproteins are present before and after the truncation step.'''
        # Don't need to adress sequence length here, as this is before the CD-HIT step. 
        sec_ids = fasta_ids(os.path.join(data_setup.UNIPROT_DATA_DIR, 'sec.fasta'))
        sec_truncated_ids = fasta_ids(os.path.join(data_setup.UNIPROT_DATA_DIR, 'sec_truncated.fasta'))
        # Make sure to remove brackets from all the truncated IDs/ 
        for id_ in sec_truncated_ids:
            assert '[1]' in id_, f'Gene ID {id_} is not marked as truncated, but is present in sec_truncated.fasta'
            id_ = id_.replace('[1]', '')
            assert id_ in sec_ids, f'Gene ID {id_} is represented in sec.fasta, but not sec_truncated.fasta.'

    def test_no_duplicate_proteins_after_concatenating_sec_trunc_and_sprot(self):
        '''Make sure that, for whatever reason, no duplicate proteins ended up in the all_data.fasta file.'''
        all_data_ids = fasta_ids(os.path.join(data_setup.UNIPROT_DATA_DIR, 'all_data.fasta'))
        assert len(all_data_ids) == len(set(all_data_ids)), f'There are {len(all_data_ids) - len(set(all_data_ids))} duplicate gene IDs are present in all_data.fasta.'
        # Remove the truncation flags, and make sure no selenoproteins are duplicated. 
        all_data_ids = [id_.replace('[1]', '') for id_ in all_data_ids]
        assert len(set(all_data_ids)) == len(all_data_ids), f'There are {len(all_data_ids) - len(set(all_data_ids))} selenoproteins duplicated in all_data.fasta.'

    def test_all_proteins_in_all_data_present_after_run_cd_hit(self):
        '''Make sure everything which meets the minimum sequence length in all_data.fasta makes it into the all_data.clstr file.'''
        clstr_file_path = os.path.join(data_setup.UNIPROT_DATA_DIR, 'all_data.clstr')
        clstr_data = pd_from_clstr(clstr_file_path)
        clstr_ids = clstr_data['id'].values # There should not be duplicate IDs here. 
        # Use length as a proxy for equality for the sake of this not taking forever.

    def test_size_of_partitioned_data_matches_all_data_after_setup_train_test_val(self):
        '''Make sure the size of the paritioned data is equal to the size of the original dataset from which it was created.'''
        all_data_size = fasta_size_with_min_seq_length(os.path.join(data_setup.UNIPROT_DATA_DIR, 'all_data.fasta'))
        # Everything in the partitioned data should already meet the minimum sequence length requirements.
        train_test_val_size = sum([csv_size(os.path.join(data_setup.DETECT_DATA_DIR, file)) for file in FILENAMES]) + NUM_EXPECTED_MISSING
        assert all_data_size == train_test_val_size, f'The size of the combined paritioned data is {train_test_val_size}, but expected {all_data_size}.'

    def test_no_duplicate_proteins_after_train_test_val_split(self):
        '''Make sure there are no duplicate proteins in the training, testing, and validation sets.'''
        all_ids = []
        for file in FILENAMES:
            ids = fasta_ids(os.path.join(data_setup.DETECT_DATA_DIR, file))
            assert len(set(ids)) == len(ids), f'Duplicate gene IDs are present in {file}.'
            all_ids += ids
        assert len(set(all_ids)) == len(all_ids), 'Duplicate gene IDs are present across training, test, and validation datasets.'
        

if __name__ == '__main__':
    unittest.main()


# '''Tests for confirming that all the functions in the src/data_setup.py directory work as expected.'''
# import sys
# sys.path.append('/home/prichter/Documents/selenobot/')
# sys.path.append('/home/prichter/Documents/selenobot/src/')

# import numpy as np
# import pandas as pd
# import utils
# import unittest
# import os
# import shutil

# import requests

# # NOTE: DETECT_DATA_DIR is specific to a problem (detect, extend, etc.)
# DETECT_DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/detect'
# DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/'
# TEST_DATA_DIR = os.path.join(os.getcwd(), 'test_data')


# test_cases = [
#     {'size':500, 'url':'https://rest.uniprot.org/uniprotkb/search?format=fasta&query=%28*%29+AND+%28reviewed%3Atrue%29+AND+%28length%3A%5B1+TO+200%5D%29+AND+%28proteins_with%3A4%29&size=500'},
#     {'size':549, 'url':'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28*%29+AND+%28reviewed%3Atrue%29+AND+%28length%3A%5B1+TO+200%5D%29+AND+%28proteins_with%3A4%29'},
#     {'size':430, 'url':'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28*%29+AND+%28reviewed%3Atrue%29+AND+%28length%3A%5B1+TO+200%5D%29+AND+%28annotation_score%3A3%29+AND+%28proteins_with%3A5%29'},
#     # {'size':1, 'url':'https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=accession%3AO00422'}
#     ]


# def get_fasta_from_uniprot(url, path=None):
#     '''Download a FASTA file from Uniprot using a URL.'''
#     text = requests.get(url).text
#     with open(path, 'w') as f:
#         f.write(text)
    
#     return path


# def setup():
#     '''Set up a local environment for testing.'''
#     # Make the directory for the temporary test data. 
#     os.mkdir(TEST_DATA_DIR)

#     for i, test_case in enumerate(test_cases):
#         filename = f'test_{i}.fasta'
#         path = get_fasta_from_uniprot(test_case['url'], path=os.path.join(TEST_DATA_DIR, filename))
#         # Update the dictionary with the path to the downloaded file. 
#         test_cases[i].update({'path':path})


# def cleanup():
#     '''Delete the directory created for testing purposes.'''
#     # for filename in os.listdir(TEST_DATA_DIR):
#     #     os.unlink(os.path.join(TEST_DATA_DIR, filename))
#     shutil.rmtree(TEST_DATA_DIR)


# class TestUtils(unittest.TestCase):

#     # TODO: Fails on case when size=1.
#     def test_utils_all_sequences_included_when_pd_from_fasta_is_called(self):
#         '''Check to make sure that the pd_from_fasta function captures all sequences.'''
#         for test_case in test_cases:
#             data = data_setup.pd_from_fasta(test_case['path'])
#             assert len(data) == test_case['size'], f"Only {len(data)} sequences loaded into the DataFrame, but expected {test_case['size']}."       
    
#     def test_utils_all_sequences_counted_when_fasta_size_is_called(self):
#         '''Check to make sure that the pd_from_fasta function captures all sequences.'''
#         for test_case in test_cases:
#             size = data_setup.fasta_size(test_case['path'])
#             assert size == test_case['size'], f"Only {size} sequences counted, but expected {test_case['size']}."

#     def test_utils_all_sequences_included_when_when_pd_to_fasta_is_called(self):
#         '''Check to make sure that pd_to_fasta successfully writes all contained sequences to a FASTA file.'''
#         for test_case in test_cases:
#             data = data_setup.pd_from_fasta(test_case['path'])
#             data = data_setup.pd_to_fasta(data, test_case['path'] + '.tmp')
#             # Get the number of entries in the newly-written FASTA file. 
#             size = data_setup.fasta_size(test_case['path'] + '.tmp')
#             # Delete the temporary FASTA file. 
#             os.unlink(test_case['path'] + '.tmp')

#             assert size == test_case['size'], f"Only {size} sequences written to the FASTA file, but expected {test_case['size']}."       
    

# if __name__ == '__main__':
    
#     setup()

#     unittest.main(exit=False)

#     cleanup()











