
from selenobot.utils import seed
seed(42) # Seed everything to ensure consistent behavior. 

from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--type', default='plm', type=str)
    parser.add_argument('--hidden-dim', default=512, type=int)
    parser.add_argument('--output-dim', default=2, type=int, choices=[2, 3])
    parser.add_argument('--weighted-loss', action='store_true')
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--models-dir', default='../models', type=str)
    args = parser.parse_args()

    # input_dims = {'plm':1024, 'aac':21, 'len':1} # Dimension of the input embeddings for each feature type. 
    model_name = f'ternary_model_{args.type}.pkl' if (args.output_dim == 3) else f'binary_model_{args.type}.pkl' 


    train_dataset = Dataset.from_hdf(os.path.join(args.data_dir, 'train.h5'), feature_type=args.type, n_classes=args.output_dim)
    val_dataset = Dataset.from_hdf(os.path.join(args.data_dir, 'val.h5'), feature_type=args.type, n_classes=args.output_dim)
    model = Classifier(n_classes=args.output_dim, input_dim=train_dataset.shape()[-1])
    print('Loaded training and validation datasets.')

    kwargs = dict()
    kwargs['balance_batches'] = False if args.weighted_loss else True
    kwargs['weighted_loss'] = args.weighted_loss 
    kwargs['lr'] = 1e-8 
    # kwargs['lr'] = 1e-6 
    kwargs['batch_size'] = 16
    kwargs['epochs'] = args.epochs

    model.fit(train_dataset, val_dataset, **kwargs)
    model.save(os.path.join(args.models_dir, model_name))

    print(f'Model training complete. Model data saved to {os.path.join(args.models_dir, model_name)}')

