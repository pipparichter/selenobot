import sys
sys.path.append('/home/prichter/Documents/selenobot')

import unittest
from utils.data import fasta_to_df, read

import numpy as np
import pandas as pd


data_dir = '/home/prichter/Documents/selenobot/data/'

class TestData(unittest.TestCase):
    '''A series of tests to check the validity of the data stored in 
    selenobot/data.'''

    def test_uniprot_sec_trunc(self):
        '''Make sure the truncated selenoprotein data matches up with
        the original selenoprotein data.'''

        # Load both files into pandas DataFrames. 
        df_sec = fasta_to_df(read(data_dir + 'uniprot_081123_sec.fasta'))
        df_sec_trunc = fasta_to_df(read(data_dir + 'uniprot_081123_sec_trunc.fasta'))

        # There should be one entry for every selenocysteine present in a sequence, for
        # each sequence in the selenoprotein database.
        df_sec['sec_counts'] = df_sec['seq'].apply(lambda s : s.count('U'))
        assert len(df_sec_trunc) == sum(df_sec['sec_counts'].values)

        # Every gene ID in the df_sec dataset is represented in the truncated dataset. 
        assert set(df_sec.index) == set(df_sec_trunc.index)

        # The number of entries for a gene ID in the df_sec_trunc should correspond
        # to the value in the df_sec sec_counts column. 
        assert np.all([df_sec['sec_counts'].loc[id_] == np.sum(df_sec_trunc.index == id_) for id_ in df_sec.index])
        



if __name__ == '__main__':
    unittest.main()

