from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import os




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', default='plm', type=str)
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--n-classes', default=2, type=int, choices=[2, 3])
    parser.add_argument('--models-dir', default='../models', type=str)
    parser.add_argument('--results-dir', default='.', type=str)

    args = parser.parse_args()

    model_name = f'ternary_model_{args.feature_type}.pkl' if (args.n_classes == 3) else f'binary_model_{args.feature_type}.pkl' 
    print(model_name)
    model_path = os.path.join(args.models_dir, model_name)

    model = Classifier.load(model_path)
    
    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, n_classes=args.n_classes)

    output_file_name, _ = os.path.splitext(os.path.basename(args.input_path))
    output_file_name = output_file_name + f"_predictions_{model_name.replace('.pkl', '')}.csv"
    output_path = os.path.join(args.results_dir, output_file_name)

    results = dataset.metadata
    predictions = model.predict(dataset)
    results = results.merge(predictions, left_index=True, right_index=True)
    results.to_csv(output_path)

    print(f'Predictions written to {output_path}')


