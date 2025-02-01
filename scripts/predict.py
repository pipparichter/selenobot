from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import os
from selenobot.utils import default_output_path


def predict(model_type:str, feature_type:str, model_name_format='{model_type}_model_{feature_type}', models_dir:str=None, input_path:str=None) -> pd.DataFrame:
    '''Load a model and dataset for the specified model type and feature type, and generate predictions for the dataset.'''
    model_name = model_name_format.format(model_type=model_type, feature_type=feature_type)
    model = Classifier.load(os.path.join(models_dir, model_name + '.pkl'))
    dataset = Dataset.from_hdf(input_path, feature_type=feature_type, n_classes=3 if (model_type == 'ternary') else 2)
    results_df = model.predict(dataset)
    results_df = results_df.rename(columns={col:f'{model_name}_{col}' for col in results_df.columns})
    return results_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-types', nargs='+', default=['plm_esm_gap', 'plm_esm_cls', 'plm_esm_log', 'plm_pt5', 'aa_1mer', 'len'])
    parser.add_argument('--model-types', nargs='+', default=['binary'])
    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--models-dir', default='../models', type=str)
    parser.add_argument('--results-dir', default='../data/results/', type=str)

    args = parser.parse_args()

    output_path = default_output_path(os.path.basename(args.input_path), op='predict', ext='csv')
    output_path = os.path.join(args.results_dir, output_path)

    results_df = list()
    pbar = tqdm(total=len(args.feature_types) * len(args.model_types), desc='predict')
    for model_type in args.model_types:
        for feature_type in args.feature_types:
            pbar.set_description(f'predict: Predicting using {model_type} model trained on {feature_type} features.')
            results_df.append(predict(model_type, feature_type, models_dir=args.models_dir, input_path=args.input_path))
            pbar.update(1)

    results_df = pd.concat(results_df, axis=1)
    metadata_df = pd.read_hdf(args.input_path, key='metadata')
    results_df = results_df.merge(metadata_df, left_index=True, right_index=True) # Add the metadata to the results.
    results_df.to_csv(output_path)

    print(f'predict: Predictions written to {output_path}')


