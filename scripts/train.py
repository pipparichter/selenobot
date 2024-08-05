from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import logging
from selenobot.utils import DATA_DIR, WEIGHTS_DIR
import os 

# Set up a log file for the training process. Also write stuff to terminal. 
logging.basicConfig(filename='/home/prichter/Documents/selenobot/train.log', level=logging.INFO, force=True, format='%(message)s')

TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
VAL_PATH = os.path.join(DATA_DIR, 'val.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='The number of epochs for which to train the model.')
    parser.add_argument('--lr', default=0.01, type=float, help='The learning rate for training the model.')
    parser.add_argument('--batch-size', default=16, type=int, help='The size of batches used to train the model.')
    parser.add_argument('--file-name', default='model.json', help='The path where the model training info will be saved.')
    args = parser.parse_args()

    model = Classifier(input_dim=1024, hidden_dim=512)

    train_dataset = Dataset(pd.read_csv(TRAIN_PATH, index_col=0))
    test_dataset = Dataset(pd.read_csv(TEST_PATH, index_col=0))
    val_dataset = Dataset(pd.read_csv(VAL_PATH, index_col=0))
    print('Loaded training, testing, and validation datasets.')

    model.fit(train_dataset, val_dataset, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)
    model.save(os.path.join(WEIGHTS_DIR, args.file_name))

