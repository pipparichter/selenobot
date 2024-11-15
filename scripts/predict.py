from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import os




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', default='plm', type=str)
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--n-classes', default=2)
    parser.add_argument('--models-dir', default='../models', type=str)
    parser.add_argument('--results-dir', default='../results', type=str)

    args = parser.parse_args()

    model_name = f'ternary_model_{args.feature_type}.pkl' if (args.n_classes == 3) else f'binary_model_{args.feature_type}.pkl' 
    model_path = os.path.join(args.model_dir, model_name)

    if args.n_classes == 2:
        dataset = BinaryDataset.from_hdf(args.input_path, feature_type=args.feature_type)
        model = BinaryClassifier.load(model_path)
    elif args.output_dim == 3:
        dataset = TernaryDataset.from_hdf(args.input_path, feature_type=args.feature_type)
        model = TernaryClassifier.load(model_path)

    output_file_name, _ = os.path.splitext(os.path.basename(args.input_path))
    output_file_name = output_file_name + '_predictions.csv'
    output_path = os.path.join(args.results_dir, output_file_name)

    results = dataset.metadata
    results['prediction'] = model.predict(dataset)
    results.to_csv(output_path)

    print(f'Predictions written to {output_path}')


