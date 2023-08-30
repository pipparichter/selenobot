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


    def test_remove_sec_overlap_from_sprot(self):
        '''Ensuring no overlap between selenoproteins and representative SwissProt sequences.'''

        df_sec = fasta_to_df(read(sec_path))
        df_sprot = fasta_to_df(read(sprot_filtered_path))

        sec_ids = df_sec.index.to_numpy()
        overlap = sec_ids[np.isin(sec_ids, df_sprot.index)]

        msg = 'The following sequences are present in both the filtered SwissProt and selenoprotein FASTA files:\n' + '\n'.join(overlap)
        assert len(overlap) > 0, msg
        

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

    def test_all(self):
        
        df_all = fasta_to_df(read(data_dir + 'all.fasta'))

        # Want to make sure every gene ID flagged as truncated has a full-length counterpart. 
        df_all_trunc = get_truncated_from_df(df_all)
        assert np.all(df_all_trunc.index.isin(df_all.index))



if __name__ == '__main__':
    unittest.main()

