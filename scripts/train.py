from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
from selenobot.utils import DATA_DIR, MODELS_DIR, ROOT_DIR
import os

# sbatch --time 10:00:00 --mem 100GB --gres gpu:1 --partition gpu --wrap "python train.py"
# srun --time 10:00:00 --mem 100GB --gres gpu:1 --partition gpu python train.py

# Set up a log file for the training process. Also write stuff to terminal. 
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_PATH = os.path.join(DATA_DIR, 'val.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='The number of epochs for which to train the model.')
    parser.add_argument('--lr', default=1e-8, type=float, help='The learning rate for training the model.')
    parser.add_argument('--batch-size', default=16, type=int, help='The size of batches used to train the model.')
    parser.add_argument('--file-name', default='model.pkl', help='The path where the model training info will be saved.')
    parser.add_argument('--n-features', default=None, type=int)
    parser.add_argument('--hidden-dim', default=512, type=int, help='The number of nodes in the hidden layer of the model.')
    parser.add_argument('--half-precision', default=False, type=bool, help='Whether or not to use half-precision floats during model training.')
    args = parser.parse_args()

    train_dataset = Dataset(pd.read_csv(TRAIN_PATH, index_col=0), n_features=args.n_features, half_precision=args.half_precision)
    val_dataset = Dataset(pd.read_csv(VAL_PATH, index_col=0), n_features=args.n_features, half_precision=args.half_precision)
    print('Loaded training, testing, and validation datasets.')
    
    model = Classifier(input_dim=train_dataset.shape()[-1], hidden_dim=args.hidden_dim, half_precision=args.half_precision)

    model.fit(train_dataset, val_dataset, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    model.save(os.path.join(MODELS_DIR, args.file_name))

    print(f'Model training complete. Model data saved to {os.path.join(MODELS_DIR, args.file_name)}')
    print('Final training loss:', model.train_losses[-1])
    print('Best validation accuracy:', max(model.val_accs))

