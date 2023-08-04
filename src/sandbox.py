import numpy as np
import h5py
import pandas as pd

figure_dir = '/home/prichter/Documents/selenobot/figures/'
data_dir = '/home/prichter/Documents/selenobot/data/'


def h5_to_df(filename):
    '''
    Load an EmbeddingDataset object from a .h5 file.
    '''

    file_ = h5py.File(filename)
    data = [] 

    gene_names = list(file_.keys())

    for key in gene_names:
        # What does this do?
        data.append(file_[key][()])
    
    file_.close()

    data = np.array(data)

    df = pd.DataFrame(data)
    df['id'] = gene_names
    df = df.astype({'id':'string'})
    
    return df


# df1 = h5_to_df(data_dir + 'test.28Jul2023.embeddings.h5')

# df2 = pd.read_csv(data_dir + 'test.csv', usecols=['label', 'id']) #, index_col='id')
# df2 = df2.astype({'id':'string'})

# df = df1.merge(df2, on='id', how='left')
# # Make sure the merge worked as intended. 
# assert (len(df1) == len(df)) and (len(df2) == len(df))

# df = df.set_index('id')
# df.to_csv(data_dir + 'test_embeddings_pr5.csv')

# df = pd.read_csv(data_dir + 'train.csv', index_col=0)
# df = df.set_index('id')
# df.to_csv(data_dir + 'train.csv')



