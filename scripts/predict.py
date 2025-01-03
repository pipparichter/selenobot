from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import os




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='plm', type=str)
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--n-classes', default=2, type=int, choices=[2, 3])
    parser.add_argument('--models-dir', default='../models', type=str)
    parser.add_argument('--results-dir', default='../data/results/', type=str)
    parser.add_argument('--add-metadata', action='store_true')

    args = parser.parse_args()

    model_name = f'ternary_model_{args.type}.pkl' if (args.n_classes == 3) else f'binary_model_{args.type}.pkl' 
    model_path = os.path.join(args.models_dir, model_name)

    model = Classifier.load(model_path)
    
    dataset = Dataset.from_hdf(args.input_path, feature_type=args.type, n_classes=args.n_classes)

    results_file_name, _ = os.path.splitext(os.path.basename(args.input_path))
    results_file_name = 'predict_' + results_file_name + f"_{model_name.replace('.pkl', '')}.csv"
    results_path = os.path.join(args.results_dir, results_file_name)

    results_df = model.predict(dataset)
    if args.add_metadata:
        results_df = results_df.merge(dataset.metadata, left_index=True, right_index=True)
    results_df.to_csv(results_path)

    print(f'Predictions written to {results_path}')


