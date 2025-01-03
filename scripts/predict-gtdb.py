import os
import re
from tqdm import tqdm 
from fabapi import * 
from selenobot.files import EmbeddingsFile
# from selenobot.utils import MODELS_DIR
from selenobot.classifiers import Classifier
import argparse
import numpy as np 
import pandas as pd
from selenobot.datasets import Dataset 
from typing import List 

# Define some important directories...
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results') # Get the path where results are stored.
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
DATA_DIR = os.path.join(ROOT_DIR, 'data') # Get the path where results are stored. 
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts') # Get the path where results are stored.

EMBEDDINGS_DIR = '/central/groups/fischergroup/prichter/gtdb/embeddings'

# Count queries are kind of slow, so just storing the table sizes as variables. 
PROTEINS_TABLE_SIZE = 200505361
ANNOTATIONS_KEGG_TABLE_SIZE = 98265928

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8315937/
#   seld_copy_num (annotations_kegg table) K01008
#       Selenocysteine is generated through a synthesis pathway where inorganic phosphate reacts with hydrogen selenide to generate 
#       selenophosphate by the selenophosphate synthetase, SelD
#   sela_copy_num (annotations_kegg table) K01042
#       Through the use of a selenocysteinyl-tRNA (Sec) synthase, SelA, selenophosphate is incorporated into serine-charged tRNAs to 
#       generate selenocysteine.
#   selb_copy_num (annotations_kegg table) K03833
#       The selenocysteine-specific elongation factor, SelB, recognizes an in-frame stop codon followed by a selenocysteine insertion 
#       sequence (SECIS).

SELD_KO = 'K01008' 
SELA_KO = 'K01042' 
SELB_KO = 'K03833' 

# sbatch --mem 100GB --time 100:00:00 --wrap "python predict-gtdb.py --results-dir /central/groups/fischergroup/prichter/selenobot/results/"
def get_copy_numbers(output_path:str=None):
    '''Take what is probably a faster approach to finding copy numbers, which is to grab every annotation which matches one of the 
    selenoprotein genes, and then group the result by genome.'''
    query = Query('annotations_kegg')
    query.equal_to('ko', [SELA_KO, SELB_KO, SELD_KO])
    results_dir, _ = os.path.split(output_path)

    total = query.count() 
    print(f'get_copy_numbers: {total} genes annotated as selA, selB, or selD.')

    page_df = query.next()
    selabd_annotations_df = []
    pbar = tqdm(total=total, desc='get_copy_numbers: Retrieving selenoprotein gene copy numbers... (page 0)')
    while page_df is not None:
        selabd_annotations_df.append(page_df)
        pbar.update(len(page_df))
        pbar.set_description(f'get_copy_numbers: Retrieving selenoprotein gene copy numbers... (page {len(selabd_annotations_df)})')
        page_df = query.next() 
    selabd_annotations_df = pd.concat(selabd_annotations_df)
    # Save the annotation results as an intermediate. 
    selabd_annotations_df.set_index('id').to_csv(os.path.join(results_dir, 'gtdb_selabd_annotations.csv'))

    copy_nums_df = []
    for genome_id, genome_id_df in selabd_annotations_df.groupby('genome_id'):
        row = dict()
        row['seld_copy_num'] = np.sum(genome_id_df.ko == SELD_KO).item()
        row['sela_copy_num'] = np.sum(genome_id_df.ko == SELA_KO).item()
        row['selb_copy_num'] = np.sum(genome_id_df.ko == SELB_KO).item()
        row['genome_id'] = genome_id
        copy_nums_df.append(row)
    copy_nums_df = pd.DataFrame(copy_nums_df)

    copy_nums_df = pd.DataFrame(copy_nums_df).set_index('genome_id')
    copy_nums_df.to_csv(os.path.join(RESULTS_DIR, 'gtdb_copy_nums.csv'))
    print(f"get_copy_numbers: Copy number information written to {output_path}")


def get_genome_data(genome_ids:List[str], batch_size=50, output_path:str=None):
    '''Retrieve the count of selenocysteine tRNAs in the genome.'''
    genome_data_df = []
    for batch in tqdm([genome_ids[i * batch_size:(i + 1) * batch_size] for i in range(len(genome_ids) // batch_size + 1)], desc='get_genome_data'):
        query = Query('metadata')
        query.equal_to('genome_id', batch)
        genome_data_df.append(query.get())
    genome_data_df = pd.concat(genome_data_df).set_index('genome_id')
    # Don't want the GC content duplicated, as this is also a field in the proteins table. 
    genome_data_df = genome_data_df.drop(columns=['gc_content', 'version'])
    genome_data_df.to_csv(output_path)
    print(f"get_genome_data: Genome data written to {output_path}")

# def get_genome_data_all(output_path:str=None):
#     '''Retrieve metadata for all genomes.'''
#     genome_data_df = []
#     query = Query('metadata')
#     page_df = query.next()
#     count, total = 0, query.count()
#     pbar = tqdm(total=total, desc='get_genome_data_all')
#     while count < total:
#         pbar.update(len(page_df))
#         count += len(page_df)
#         genome_data_df.append(page_df)
#         page_df = query.next()
#     genome_data_df = pd.concat(genome_data_df).set_index('genome_id')
#     genome_data_df.to_csv(output_path)
#     print(f"get_genome_data_all: Genome data for all genomes written to {output_path}") 


def get_annotation_data(ids:List[str], batch_size=100, output_path:str=None):
    annotation_data_df = []
    for batch in tqdm([ids[i * batch_size:(i + 1) * batch_size] for i in range(len(ids) // batch_size + 1)], desc='get_annotation_data: Fetching annotations...'):
        query = Query('annotations_kegg')
        query.equal_to('gene_id', batch)
        # Not every gene has an annotation, so need to catch that case.
        batch_df = query.get()
        if batch_df is not None:
            annotation_data_df.append(query.get()[['ko', 'gene_id']])
    annotation_data_df = pd.concat(annotation_data_df)
    annotation_data_df = annotation_data_df.rename(columns={'gene_id':'id'})
    annotation_data_df.set_index('id').to_csv(os.path.join(output_path))
    print(f"get_annotation_data: Annotation data written to {output_path}")


def get_gene_data(ids:List[str], batch_size=100, output_path:str=None):
    # NOTE: As long as the batch size is less than 1000 (which I think is the default page size), should not need to paginate at all. 
    gene_data_df = []
    for batch in tqdm([ids[i * batch_size:(i + 1) * batch_size] for i in range(len(ids) // batch_size + 1)], desc='get_gene_data: Fetching gene data...'):
        query = Query('proteins')
        query.equal_to('gene_id', batch)
        gene_data_df.append(query.get())
    gene_data_df = pd.concat(gene_data_df)
    gene_data_df = gene_data_df.rename(columns={'gene_id':'id'})
    gene_data_df.set_index('id').to_csv(output_path)
    print(f"get_gene_data: Gene and annotations data written to {output_path}")


def get_predictions(model:str, embeddings_dir:str=EMBEDDINGS_DIR, models_dir:str=MODELS_DIR, output_path:str=None):
    '''Run the trained model on all embedded genomes in the embeddings directory.'''

    model = Classifier.load(os.path.join(models_dir, model)) # Load the pre-trained model. 
    print(f'get_predictions: Loaded model {args.model}.')

    embeddings_file_names = os.listdir(embeddings_dir) # Filenames are the RS_ or GB_ prefix followed by the genome ID. 
    genome_ids = [re.search(r'GC[AF]_\d{9}\.\d{1}', file_name).group(0) for file_name in embeddings_file_names]

    predictions_df = []

    for embeddings_file_name, genome_id in tqdm(zip(embeddings_file_names, genome_ids), total=len(embeddings_file_names), desc='get_predictions: Processing genomes...'):
        
        embeddings_file = EmbeddingsFile(os.path.join(embeddings_dir, embeddings_file_name))
            
        dataset = Dataset(embeddings_file.to_df().set_index('id')) # Instantiate a Dataset object with the embeddings. 
        predictions_raw = model.predict(dataset, threshold=None).ravel()
        predictions_threshold = np.array([1 if p > 0.5 else 0 for p in predictions_raw])
        df = pd.DataFrame({'id':dataset.ids, 'model_output':predictions_raw, 'prediction':predictions_threshold})
        df = df[df.prediction == 1] # Filter for the predicted selenoproteins. 
        # print(f'get_predictions: {len(df)} predicted selenoproteins in genome {genome_id}.')

        predictions_df.append(df)

    predictions_df = pd.concat(predictions_df).set_index('id')
    predictions_df.to_csv(output_path)
    print(f"get_predictions: Predicted selenoproteins written to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model_epochs_1000_lr_e8.pkl', type=str)
    # parser.add_argument('--output-path', default=os.path.join(RESULTS_DIR, 'gtdb_results.csv'), type=str)

    args = parser.parse_args()
    print('Using results directory', RESULTS_DIR)

    # get_genome_data_all(os.path.join(RESULTS_DIR, 'gtdb_genome_data_all.csv'))
    
    if not os.path.exists(os.path.join(RESULTS_DIR, 'gtdb_predictions.csv')):
        get_predictions(args.model, output_path=os.path.join(RESULTS_DIR, 'gtdb_predictions.csv'))
    predictions_df = pd.read_csv(os.path.join(RESULTS_DIR, 'gtdb_predictions.csv'))
    
    if not os.path.exists(os.path.join(RESULTS_DIR, 'gtdb_gene_data.csv')):
        get_gene_data(predictions_df['id'].values, output_path=os.path.join(RESULTS_DIR, 'gtdb_gene_data.csv'))   
    gene_data_df = pd.read_csv(os.path.join(RESULTS_DIR, 'gtdb_gene_data.csv'), dtype={'partial':str})
    
    if not os.path.exists(os.path.join(RESULTS_DIR, 'gtdb_annotation_data.csv')):
        get_annotation_data(predictions_df['id'].values, output_path=os.path.join(RESULTS_DIR, 'gtdb_annotation_data.csv'))   
    annotation_data_df = pd.read_csv(os.path.join(RESULTS_DIR, 'gtdb_annotation_data.csv'), dtype={'partial':str})

    if not os.path.exists(os.path.join(RESULTS_DIR, 'gtdb_genome_data.csv')):
        get_genome_data(gene_data_df.genome_id.unique(), output_path=os.path.join(RESULTS_DIR, 'gtdb_genome_data.csv'))
    genome_data_df = pd.read_csv(os.path.join(RESULTS_DIR, 'gtdb_genome_data.csv')) # , index_col=0)

    if not os.path.exists(os.path.join(RESULTS_DIR, 'gtdb_copy_nums.csv')):
        # Use genome IDs from the stop_codons_df, as genome IDs are not included in the predictions_df.   
        get_copy_numbers(output_path=os.path.join(RESULTS_DIR, 'gtdb_copy_nums.csv')) 
    copy_nums_df = pd.read_csv(os.path.join(RESULTS_DIR, 'gtdb_copy_nums.csv')) # , index_col=0)


    results_df = predictions_df.merge(gene_data_df, how='left', left_on='id', right_on='id')

    # annotated_ids = annotation_data_df.id.unique()
    # annotations = []
    # for id in tqdm(results_df.id, 'Adding annotations to results...'):
    #     if id in annotated_ids:
    #         annotations.append(','.join(list(annotation_data_df[annotation_data_df.id == id].ko)))
    #     else:
    #         annotations.append('')
    # results_df['ko'] = annotations 

    results_df = results_df.merge(annotation_data_df, how='left', left_on='id', right_on='id')
    results_df = results_df.merge(copy_nums_df, how='left', left_on='genome_id', right_on='genome_id')
    results_df = results_df.merge(genome_data_df, how='left', left_on='genome_id', right_on='genome_id')

    output_path = os.path.join(RESULTS_DIR, 'gtdb_results.csv')
    results_df = results_df.set_index('id')
    results_df.to_csv(output_path)
    print(f'Results written to {output_path}')



# def get_copy_numbers(genome_id:str):
#     '''Retrieve the copy numbers of genes related to selenoprotein synthesis for the genome.'''
#     query = GetQuery('annotations_kegg')
#     query.equal_to('genome_id', genome_id)
#     query.equal_to('ko', [SELA_KO, SELB_KO, SELD_KO])

#     page_df = query.next()
#     df = []
#     while page_df is not None:
#         df.append(page_df)
#         page_df = query.next()

#     if len(df) == 0:
#         print(f'get_copy_numbers: No selA, selB, or selD found in genome {genome_id}.')
#         return 0, 0, 0
#     else:
#         df = pd.concat(df)
#         seld_copy_num = len(df.ko == SELD_KO)
#         sela_copy_num = len(df.ko == SELA_KO)
#         selb_copy_num = len(df.ko == SELB_KO)
#         return seld_copy_num, sela_copy_num, selb_copy_num















