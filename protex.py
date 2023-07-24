from prettytable import PrettyTable
import pandas as pd 
import numpy as np

def summary(model):
    '''
    Print out a summary of the model weights, and which parameters are trainable. 
    '''
    table = PrettyTable(['name', 'num_params', 'fixed'])

    num_fixed = 0
    num_total = 0

    params = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        fixed = str(not param.requires_grad)
        table.add_row([name, num_params, fixed])

        if not param.requires_grad:
            num_fixed += num_params
        
        num_total += num_params
    
    print(table)
    print('TOTAL:', num_total)
    print('TRAINABLE:', num_total - num_fixed, f'({int(100 * (num_total - num_fixed)/num_total)}%)')

if __name__ == '__main__':
    # Probably should organize the embeddings and get rid of the duplicates. 
    encodings = pd.read_csv('./data/test_embeddings.csv', header=None).values
    indices = pd.read_csv('./data/test_indices.csv', header=None).values

    data = pd.DataFrame(encodings, columns=[str(i) for i in range(encodings.shape[1])])
    data['index'] = indices
    data = data.drop_duplicates('index')

    data.to_csv('./data/test_embeddings_01.csv', index=False)
