
from selenobot.utils import seed
seed(42)

from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--feature-type', default='plm', type=str, choices=['plm', 'aac', 'len'])
    parser.add_argument('--hidden-dim', default=512, type=int)
    parser.add_argument('--output-dim', default=2, type=int, choices=[2, 3])
    parser.add_argument('--weighted-loss', action='store_true')
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--models-dir', default='../models', type=str)
    args = parser.parse_args()

    model_name = f'ternary_model_{args.feature_type}.pkl' if (args.output_dim == 3) else f'binary_model_{args.feature_type}.pkl' 

    train_dataset = Dataset.from_hdf(os.path.join(args.data_dir, 'train.h5'), n_classes=args.output_dim, feature_type=args.feature_type)
    val_dataset = Dataset.from_hdf(os.path.join(args.data_dir, 'val.h5'), n_classes=args.output_dim, feature_type=args.feature_type)
    print('Loaded training and validation datasets.')

    model = TernaryClassifier(input_dim=1024, hidden_dim=args.hidden_dim) if (args.output_dim == 3) else BinaryClassifier(input_dim=1024, hidden_dim=args.hidden_dim)

    kwargs = dict()
    kwargs['batch_size']
    kwargs['balance_batches'] = False if args.weighted_loss else True
    kwargs['weighted_loss'] = args.weighted_loss 
    kwargs['lr'] = 1e-8 
    kwargs['batch_size'] = 16
    kwargs['epochs'] = args.epochs

    model.fit(train_dataset, val_dataset, **kwargs)
    model.save(os.path.join(args.models_dir, model_name))

    print(f'Model training complete. Model data saved to {os.path.join(args.models_dir, model_name)}')

