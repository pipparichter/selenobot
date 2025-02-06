
from selenobot.utils import seed
seed(42) # Seed everything to ensure consistent behavior. 

from selenobot.classifiers import Classifier
from selenobot.datasets import Dataset 
import subprocess
import argparse
import os

# python train.py --model-name model_2c_plm_esm_log_aa_tokens_only --feature-type plm_esm_log --aa-tokens-only
# python train.py --model-name model_2c_plm_esm_cls_add_length_feature --feature-type plm_esm_cls --add-length-feature

def check(model:Classifier):
    '''Check to make sure model training worked correctly, in which case all of these attributes (and some others),
    should be populated with values.'''
    attrs = ['epochs', 'best_epoch', 'lr', 'train_accs', 'val_accs']
    for attr in attrs:
        assert hasattr(model.model, attr), f'check: Model is missing attribute {attr}, which should have been populated during training.'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', default='../data/train.h5', type=str)
    parser.add_argument('--val-data-path', default='../data/val.h5', type=str)
    parser.add_argument('--model-name', default=None, type=str)

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--feature-type', default='plm_pt5', type=str)
    parser.add_argument('--output-dim', default=2, type=int, choices=[2, 3])
    parser.add_argument('--weighted-loss', action='store_true')
    parser.add_argument('--add-length-feature', action='store_true')
    parser.add_argument('--aa-tokens-only', action='store_true')

    parser.add_argument('--models-dir', default='../models', type=str)

    args = parser.parse_args()

    model_name = f'model_{args.output_dim}c_{args.feature_type}.pkl' if (args.model_name is None) else f'{args.model_name}.pkl'
    
    train_dataset = Dataset.from_hdf(args.train_data_path, n_classes=args.output_dim, feature_type=args.feature_type, add_length_feature=args.add_length_feature, aa_tokens_only=args.aa_tokens_only)
    val_dataset = Dataset.from_hdf(args.val_data_path, n_classes=args.output_dim, feature_type=args.feature_type, add_length_feature=args.add_length_feature, aa_tokens_only=args.aa_tokens_only)

    model = Classifier(n_classes=args.output_dim, input_dim=train_dataset.shape()[-1])
    print('Loaded training and validation datasets.')

    kwargs = dict()
    kwargs['balance_batches'] = False if args.weighted_loss else True
    kwargs['weighted_loss'] = args.weighted_loss 
    kwargs['batch_size'] = 16
    kwargs['epochs'] = args.epochs
    kwargs['lr'] = 1e-8 

    model.fit(train_dataset, val_dataset, **kwargs)
    check(model)

    model.save(os.path.join(args.models_dir, model_name))

    print(f'Model training complete. Model data saved to {os.path.join(args.models_dir, model_name)}')

