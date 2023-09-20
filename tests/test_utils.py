'''Tests for confirming that all the functions in the src/utils.py directory work as expected.'''
import sys
sys.path.append('/home/prichter/Documents/selenobot/')
sys.path.append('/home/prichter/Documents/selenobot/src/')

import numpy as np
import pandas as pd
import utils
import unittest
import os
import shutil

import requests

# NOTE: DETECT_DATA_DIR is specific to a problem (detect, extend, etc.)
DETECT_DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/detect'
DATA_DIR = '/home/prichter/Documents/selenobot/data/uniprot_2023_03/'
TEST_DATA_DIR = os.path.join(os.getcwd(), 'test_data')


test_cases = [
    {'size':500, 'url':'https://rest.uniprot.org/uniprotkb/search?format=fasta&query=%28*%29+AND+%28reviewed%3Atrue%29+AND+%28length%3A%5B1+TO+200%5D%29+AND+%28proteins_with%3A4%29&size=500'},
    {'size':549, 'url':'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28*%29+AND+%28reviewed%3Atrue%29+AND+%28length%3A%5B1+TO+200%5D%29+AND+%28proteins_with%3A4%29'},
    {'size':430, 'url':'https://rest.uniprot.org/uniprotkb/stream?format=fasta&query=%28*%29+AND+%28reviewed%3Atrue%29+AND+%28length%3A%5B1+TO+200%5D%29+AND+%28annotation_score%3A3%29+AND+%28proteins_with%3A5%29'},
    # {'size':1, 'url':'https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=accession%3AO00422'}
    ]


def get_fasta_from_uniprot(url, path=None):
    '''Download a FASTA file from Uniprot using a URL.'''
    text = requests.get(url).text
    with open(path, 'w') as f:
        f.write(text)
    
    return path


def setup():
    '''Set up a local environment for testing.'''
    # Make the directory for the temporary test data. 
    os.mkdir(TEST_DATA_DIR)

    for i, test_case in enumerate(test_cases):
        filename = f'test_{i}.fasta'
        path = get_fasta_from_uniprot(test_case['url'], path=os.path.join(TEST_DATA_DIR, filename))
        # Update the dictionary with the path to the downloaded file. 
        test_cases[i].update({'path':path})


def cleanup():
    '''Delete the directory created for testing purposes.'''
    # for filename in os.listdir(TEST_DATA_DIR):
    #     os.unlink(os.path.join(TEST_DATA_DIR, filename))
    shutil.rmtree(TEST_DATA_DIR)


class TestUtils(unittest.TestCase):

    # TODO: Fails on case when size=1.
    def test_utils_all_sequences_included_when_pd_from_fasta_is_called(self):
        '''Check to make sure that the pd_from_fasta function captures all sequences.'''
        for test_case in test_cases:
            data = utils.pd_from_fasta(test_case['path'])
            assert len(data) == test_case['size'], f"Only {len(data)} sequences loaded into the DataFrame, but expected {test_case['size']}."       
    
    def test_utils_all_sequences_counted_when_fasta_size_is_called(self):
        '''Check to make sure that the pd_from_fasta function captures all sequences.'''
        for test_case in test_cases:
            size = utils.fasta_size(test_case['path'])
            assert size == test_case['size'], f"Only {size} sequences counted, but expected {test_case['size']}."

    def test_utils_all_sequences_included_when_when_pd_to_fasta_is_called(self):
        '''Check to make sure that pd_to_fasta successfully writes all contained sequences to a FASTA file.'''
        for test_case in test_cases:
            data = utils.pd_from_fasta(test_case['path'])
            data = utils.pd_to_fasta(data, test_case['path'] + '.tmp')
            # Get the number of entries in the newly-written FASTA file. 
            size = utils.fasta_size(test_case['path'] + '.tmp')
            # Delete the temporary FASTA file. 
            os.unlink(test_case['path'] + '.tmp')

            assert size == test_case['size'], f"Only {size} sequences written to the FASTA file, but expected {test_case['size']}."       
    

if __name__ == '__main__':
    
    setup()

    unittest.main(exit=False)

    cleanup()









