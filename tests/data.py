import sys
sys.path.append('/home/prichter/Documents/selenobot')
sys.path.append('/home/prichter/Documents/selenobot/src')
sys.path.append('/home/prichter/Documents/selenobot/utils')

import unittest
from utils.data import fasta_to_df, read

import numpy as np
import pandas as pd

data_dir = '/home/prichter/Documents/selenobot/data/'
sec_trunc_path = data_dir + 'uniprot_081123_sec_trunc.fasta'
sec_path = data_dir + 'uniprot_081123_sec.fasta'
sprot_filtered_path = data_dir + 'uniprot_081623_sprot_filtered.fasta'


def get_truncated_from_df(df):
    '''Accepts a pandas DataFrame of sequences. It filters out the
    DataFrame entries which are flagged with an asterisk, indicating that they
    are truncated selenoproteins.'''
    
    assert df.index.name == 'id'

    f = np.vectorize(lambda x : '*' in x)
    df = df[f(df.index.to_numpy())]
    
    return df


class TestData(unittest.TestCase):
    '''A series of tests to check the validity of the data stored in 
    selenobot/data.'''

    def test_train_test_val_split(self):
        '''Confirm that splitting the data into training, validation, 
        and test sets worked as intended. Assumes an intended train-test-validation
        split of 0.8-0.1-0.1.'''

        # Want to confirm no overlap between each dataset. 
        # Want 
        pass

    def test_truncate_sec(self):
        '''Validating truncated selenoprotein data.'''

        # Load both files into pandas DataFrames. 
        df_sec = fasta_to_df(read(sec_path))
        df_sec_trunc = fasta_to_df(read(sec_trunc_path))
        
        assert df_sec.index.name == df_sec_trunc.index.name == 'id'

        msg = f'One or both index data types ({df_sec.index.dtype} and {df_sec_trunc.index.dtype}) are incorrect.'
        assert df_sec.index.dtype == df_sec_trunc.index.dtype == object, msg

        # msg = f'df_sec and df_sec_trunc have mismatched lengths {len(df_sec)} and {len(df_sec_trunc)}.'
        # assert len(df_sec) == len(df_sec_trunc), msg

        # There should be one entry for every selenocysteine present in a sequence, for
        # each sequence in the selenoprotein database.
        df_sec['sec_counts'] = df_sec['seq'].apply(lambda s : s.count('U'))
        assert len(df_sec_trunc) == sum(df_sec['sec_counts'].values)

        # Every gene ID in the df_sec dataset is represented in the truncated dataset. 
        assert set(df_sec.index) == set([i.replace('*', '') for i in df_sec_trunc.index])

        # The number of entries for a gene ID in the df_sec_trunc should correspond
        # to the value in the df_sec sec_counts column. 
        # assert np.all([df_sec['sec_counts'].loc[id_] == np.sum(df_sec_trunc.index == id_) for id_ in df_sec.index])





if __name__ == '__main__':
    unittest.main()

