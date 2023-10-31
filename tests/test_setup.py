'''Tests to make sure the data setup worked as expected.'''
# pylint: disable=all
import unittest

import numpy as np
import pandas as pd
import os
import sys

sys.path.append('/home/prichter/Documents/selenobot/data/')
sys.path.append('/home/prichter/Documents/selenobot/src/')

import setup 
import dataset

FILENAMES = ['train.csv', 'val.csv', 'test.csv']

# TODO: Some selenoproteins will be missing because of hte CD-HIT minimum sequence length. 

# test_[method under test]_[expected behavior]?_when_[preconditions]

class TestSetup(unittest.TestCase):

    # def test_setup_all_sec_proteins_present_when_data_is_partitioned(self):
    #     '''Test to make sure all selenoproteins are accounted for in the train, test, and validation data.'''

    #     # Read in the selenoprotein ids. 
    #     ref_sec_ids = setup.fasta_ids(os.path.join(setup.UNIPROT_DATA_DIR, 'sec_truncated.fasta'))

    #     # Collect the sec IDs from the data files. 
    #     sec_ids = []
    #     for filename in FILENAMES:
    #         data = pd.read_csv(os.path.join(setup.DETECT_DATA_DIR, filename), usecols=['id', 'label'])
    #         data = data[data['label'] == 1] # Locate the IDs labeled as selenproteins. 
    #         sec_ids.append(data['id'].values)
        
    #     sec_ids = np.ravel(np.concatenate(sec_ids))
        
    #     assert len(sec_ids) == len(np.unique(sec_ids)), 'Some selenoproteins are counted multiple times in the partitioned data.'
    #     assert len(np.unique(sec_ids)) >= len(ref_sec_ids), f'There are {len(ref_sec_ids) - len(np.unique(sec_ids))} fewer selenoproteins in the datasets than in the original data.'
    #     assert np.all(np.isin(ref_sec_ids, sec_ids)), 'Not all of the selenoproteins are represented in the partitioned data.'

    def test_setup_label_is_one_when_sequence_is_truncated(self):
        '''Test to make sure all selenoproteins are flagged with a 1 in the label column.'''

        for filename in FILENAMES:
            # Can't use the actual sequence, as the U residue has been removed. 
            data = pd.read_csv(os.path.join(setup.DETECT_DATA_DIR, filename), usecols=['id', 'label'])
            for row in data.itertuples():
                if '[' in row.id:
                    assert row.label == 1, f'The selenoprotein {row.id} is mislabeled as full-length.'
    
    def test_setup_label_is_zero_when_sequence_is_not_truncated(self):
        '''Test to make sure all full-length proteins are flagged with a 0 in the label column.'''

        for filename in FILENAMES:
            # Can't use the actual sequence, as the U residue has been removed. 
            data = pd.read_csv(os.path.join(setup.DETECT_DATA_DIR, filename), usecols=['id', 'label'])
            for row in data.itertuples():
                if '[' not in row.id:
                    assert row.label == 0, f'The full-length protein {row.id} is mislabeled as truncated.'

    def test_setup_all_sec_proteins_represented_after_truncating(self):
        '''Test to make sure all selenoproteins are present before and after the truncation step.'''

        sec_ids = setup.fasta_ids(os.path.join(setup.UNIPROT_DATA_DIR, 'sec.fasta'))
        sec_truncated_ids = setup.fasta_ids(os.path.join(setup.UNIPROT_DATA_DIR, 'sec_truncated.fasta'))
        # Make sure to remove brackets from all the truncated IDs/ 
        for id_ in sec_truncated_ids:
            assert '[1]' in id_, f'Gene ID {id_} is not marked as truncated, but is present in sec_truncated.fasta'
            id_ = id_[:-3] # Remove the [1] at the end of the ID. 
            assert id_ in sec_ids, f'Gene ID {id_} is represented in sec.fasta, but not sec_truncated.fasta.'
    

    # def test_setup_all_proteins_in_all_data_represented_after_partitioning(self):
    #     '''Test to make sure no sequences were lost when partitioning data into training, testing, and validation sets.'''

    #     ref_ids = setup.fasta_ids(os.path.join(setup.UNIPROT_DATA_DIR, 'all_data.fasta'))
    #     ids = [setup.csv_ids(os.path.join(setup.DETECT_DATA_DIR, filename)) for filename in FILENAMES]
    #     ids = np.ravel(np.concatenate(ids)) # Collect all IDs in a one-dimensional numpy array. 

    #     assert np.all(np.isin(ref_ids, ids)), 'Proteins present in all_data.fasta are missing in the partitioned data. '




if __name__ == '__main__':
    unittest.main()


# '''Tests for confirming that all the functions in the src/setup.py directory work as expected.'''
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
#             data = setup.pd_from_fasta(test_case['path'])
#             assert len(data) == test_case['size'], f"Only {len(data)} sequences loaded into the DataFrame, but expected {test_case['size']}."       
    
#     def test_utils_all_sequences_counted_when_fasta_size_is_called(self):
#         '''Check to make sure that the pd_from_fasta function captures all sequences.'''
#         for test_case in test_cases:
#             size = setup.fasta_size(test_case['path'])
#             assert size == test_case['size'], f"Only {size} sequences counted, but expected {test_case['size']}."

#     def test_utils_all_sequences_included_when_when_pd_to_fasta_is_called(self):
#         '''Check to make sure that pd_to_fasta successfully writes all contained sequences to a FASTA file.'''
#         for test_case in test_cases:
#             data = setup.pd_from_fasta(test_case['path'])
#             data = setup.pd_to_fasta(data, test_case['path'] + '.tmp')
#             # Get the number of entries in the newly-written FASTA file. 
#             size = setup.fasta_size(test_case['path'] + '.tmp')
#             # Delete the temporary FASTA file. 
#             os.unlink(test_case['path'] + '.tmp')

#             assert size == test_case['size'], f"Only {size} sequences written to the FASTA file, but expected {test_case['size']}."       
    

# if __name__ == '__main__':
    
#     setup()

#     unittest.main(exit=False)

#     cleanup()











