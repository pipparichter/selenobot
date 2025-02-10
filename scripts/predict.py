from selenobot.classifiers import * 
from selenobot.datasets import * 
import subprocess
import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-type', type=str, default='plm_esm_gap')
    parser.add_argument('--n-classes', type=int, default=3)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--models-dir', default='../models', type=str)
    parser.add_argument('--results-dir', default='../data/results/', type=str)
    parser.add_argument('--add-length-feature', action='store_true')
    parser.add_argument('--aa-tokens-only', action='store_true')

    # parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    output_file_name = os.path.basename(args.input_path).replace('.h5', '.predict.csv')
    output_path = os.path.join(args.results_dir, output_file_name)
    metadata_df = pd.read_csv(output_path, index_col=0) if os.path.exists(output_path) else pd.read_hdf(args.input_path, key='metadata')

    model = Classifier.load(os.path.join(args.models_dir, args.model_name + '.pkl'))
    
    kwargs = {'add_length_feature':args.add_length_feature, 'aa_tokens_only':args.aa_tokens_only}
    dataset = Dataset.from_hdf(args.input_path, feature_type=args.feature_type, n_classes=args.n_classes, **kwargs)

    pred_df = model.predict(dataset)
    pred_df.columns = [f'{args.model_name}_{col}' for col in pred_df.columns] # Rename the columns so that the model is specified. 
    metadata_df = metadata_df.drop(columns=pred_df.columns, errors='ignore') # Drop existing predictions columns for the model.

    pred_df = pred_df.merge(metadata_df, left_index=True, right_index=True, how='left') # Add the metadata to the results.
    pred_df.to_csv(output_path)

    print(f'Predictions for model {args.model_name} written to {output_path}')


