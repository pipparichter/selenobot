
from selenobot.utils import seed
seed(42)

from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import os


# Define some important directories...
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results') # Get the path where results are stored.
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data') # Get the path where results are stored. 
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts') # Get the path where results are stored.

# sbatch --time 10:00:00 --mem 100GB --gres gpu:1 --partition gpu --wrap "python train.py"
# srun --time 10:00:00 --mem 100GB --gres gpu:1 --partition gpu python train.py

# Set up a log file for the training process. Also write stuff to terminal. 
TRAIN_PATH = os.path.join(DATA_DIR, 'train.h5')
VAL_PATH = os.path.join(DATA_DIR, 'val.h5')
TEST_PATH = os.path.join(DATA_DIR, 'test.h5')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1000, type=int, help='The number of epochs for which to train the model.')
    parser.add_argument('--lr', default=1e-8, type=float, help='The learning rate for training the model.')
    parser.add_argument('--batch-size', default=16, type=int, help='The size of batches used to train the model.')
    parser.add_argument('--feature-type', default='plm', type=str, help='The type of sequence representation to use when training the model.')
    parser.add_argument('--model-name', default='model.pkl', help='The path where the model training info will be saved.')
    parser.add_argument('--hidden-dim', default=512, type=int, help='The number of nodes in the hidden layer of the model.')
    parser.add_argument('--weighted-loss', action='store_true')
    # parser.add_argument('--early-stopping', help='Whether or not to use the model weights which performed best on the validation set.', action=argparse.BooleanOptionalAction)
    # parser.add_argument('--n-features', default=None, type=int)
    # parser.add_argument('--scale', default=True, type=bool, help='Whether or not to include a StandardScaler in the model.', action=argparse.BooleanOptionalAction)
    # parser.add_argument('--half-precision', default=True, type=bool, help='Whether or not to use half-precision floats during model training.')
    args = parser.parse_args()

    # assert args.balance_batches ^ args.weighted_loss, 'Can\'t have both balanced batches and weighted loss.'
    balance_batches = False if args.weighted_loss else True

    train_dataset = Dataset.from_hdf(TRAIN_PATH)
    val_dataset = Dataset.from_hdf(VAL_PATH)
    print('Loaded training and validation datasets.')

    # print(f"Training model with scaling {'on' if args.scale else 'off'} for {args.epochs} with learning rate {args.lr}.") 
    print(f"Training model for {args.epochs} epochs with learning rate {args.lr}.") 
    model = Classifier(input_dim=train_dataset.n_features, hidden_dim=args.hidden_dim, scale=True)

    model.fit(train_dataset, val_dataset, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, weighted_loss=args.weighted_loss, balance_batches=balance_batches)
    model.save(os.path.join(MODELS_DIR, args.model_name))

    print(f'Model training complete. Model data saved to {os.path.join(MODELS_DIR, args.model_name)}')
    print('Final training loss:', model.train_losses[-1])
    print('Best validation accuracy:', max(model.val_accs))

