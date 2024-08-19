import os
import re
from tqdm import tqdm 
from fabapi import * 
from selenobot.files import EmbeddingsFile
from selenobot.utils import MODELS_DIR, RESULTS_DIR
from selenobot.classifiers import Classifier
import argparse
import numpy as np 
import pandas as pd
from selenobot.datasets import Dataset 
from typing import List 

EMBEDDINGS_DIR = '/central/groups/fischergroup/prichter/gtdb/embeddings'

# Count queries are kind of slow, so just storing the table sizes as variables. 
PROTEINS_TABLE_SIZE = 200505361
ANNOTATIONS_KEGG_TABLE_SIZE = 98265928

# Fields to include along with the predictions:
#   sec_trna_count (metadata table)
#   stop_codon (proteins table)

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


# Failed on http://microbes.gps.caltech.edu:8000/get/proteins?gene_id[eq]JAFRYQ010000101.1_3[page]0

def get_copy_numbers():
    '''Take what is probably a faster approach to finding copy numbers, which is to grab every annotation which matches one of the 
    selenoprotein genes, and then group the result by genome.'''
    query = Query('annotations_kegg')
    query.equal_to('ko', [SELA_KO, SELB_KO, SELD_KO])

    total = query.count() 
    print(f'get_copy_numbers: {total} genes annotated as selA, selB, or selD.')

    page_df = query.next()
    df = []
    pbar = tqdm(total=total, desc='get_copy_numbers: Retrieving selenoprotein gene copy numbers... (page 0)')
    while page_df is not None:
        df.append(page_df)
        pbar.update(len(page_df))
        pbar.set_description(f'get_copy_numbers: Retrieving selenoprotein gene copy numbers... (page {len(df)})')
        page_df = query.next() 
    df = pd.concat(df)

    copy_nums_df = []
    for genome_id, genome_id_df in df.groupby('genome_id'):
        row = dict()
        row['seld_copy_num'] = len(genome_id_df.ko == SELD_KO)
        row['sela_copy_num'] = len(genome_id_df.ko == SELA_KO)
        row['selb_copy_num'] = len(genome_id_df == SELB_KO) 
        row['genome_id'] = genome_id
        copy_nums_df.append(row)

    copy_nums_df = pd.DataFrame(copy_nums_df).set_index('genome_id')
    copy_nums_df.to_csv(os.path.join(RESULTS_DIR, 'gtdb_copy_nums.csv'))
    print(f"get_copy_numbers: Copy number information written to {os.path.join(RESULTS_DIR, 'gtdb_copy_nums.csv')}")


def get_sec_trna_counts(genome_ids:List[str], batch_size=50, output_path:str=os.path.join(RESULTS_DIR, 'gtdb_sec_trna_counts.csv')):
    '''Retrieve the count of selenocysteine tRNAs in the genome.'''
    sec_trna_counts_df = []
    for batch in [gene_ids[i * batch_size:(i + 1) * batch_size] for i in range(len(genome_ids) // batch_size + 1)]:
        query = Query('metadata')
        query.equal_to('genome_id', batch)
        sec_trna_counts_df.append(query.get()[['genome_id', 'sec_trna_count']])
    
    sec_trna_counts_df = pd.concat(sec_trna_counts_df).set_index('gene_id')
    sec_trna_counts_df.to_csv(output_path)
    print(f"get_sec_trna_counts: Sec tRNA count information written to {output_path}")


def get_stop_codons(gene_ids:List[str], batch_size=50, output_path:str=os.path.join(RESULTS_DIR, 'gtdb_stop_codons.csv')):
    '''Retrieve the gene's stop codon.'''
    stop_codons_df = []
    for batch in [gene_ids[i * batch_size:(i + 1) * batch_size] for i in range(len(gene_ids) // batch_size + 1)]:
        query = Query('proteins')
        query.equal_to('gene_id', batch)
        stop_codons_df.append(query.get(print_url=True)[['gene_id', 'stop_codon']])
    
    stop_codons_df = pd.concat(stop_codons_df).set_index('gene_id')
    stop_codons_df.to_csv(output_path)
    print(f"get_stop_codons: Stop codon information written to {output_path}")


def get_predictions(model:str, embeddings_dir:str=EMBEDDINGS_DIR, models_dir:str=MODELS_DIR, output_path:str=os.path.join(RESULTS_DIR, 'gtdb_predictions.csv')):
    '''Run the trained model on all embedded genomes in the embeddings directory.'''

    model = Classifier.load(os.path.join(models_dir, model)) # Load the pre-trained model. 
    print(f'get_predictions: Loaded model {args.model}.')

    embeddings_file_names = os.listdir(embeddings_dir) # Filenames are the RS_ or GB_ prefix followed by the genome ID. 
    genome_ids = [re.search(r'GC[AF]_\d{9}\.\d{1}', file_name).group(0) for file_name in embeddings_file_names]

    predictions_df = []

    for embeddings_file_name, genome_id in tqdm(zip(embeddings_file_names, genome_ids), total=len(embeddings_file_names), desc='get_predictions: Processing genomes...'):
        
        embeddings_file = EmbeddingsFile(os.path.join(embeddings_dir, embeddings_file_name))
            
        dataset = Dataset(embeddings_file.dataframe().set_index('gene_id')) # Instantiate a Dataset object with the embeddings. 
        predictions_raw = model.predict(dataset, threshold=None)
        predictions_threshold = np.array([1 if p > 0.5 else 0 for p in predictions_raw])

        df = pd.DataFrame({'gene_id':dataset.gene_ids, 'model_output':predictions_raw, 'prediction':predictions_threshold})
        df = df[df.prediction == 1] # Filter for the predicted selenoproteins. 
        # print(f'get_predictions: {len(df)} predicted selenoproteins in genome {genome_id}.')

        predictions_df.append(df)

    predictions_df = pd.concat(predictions_df).set_index('gene_id')
    predictions_df.to_csv(output_path)
    print(f"get_predictions: Predicted selenoproteins written to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model_epochs_100_lr_e8.pkl', type=str)
    parser.add_argument('--output-path', default=os.path.join(RESULTS_DIR, 'gtdb_results.csv'), type=str)

    args = parser.parse_args()

    # if not os.path.exists(os.path.join(RESULTS_DIR, 'gtdb_predictions.csv')):
    get_predictions(args.model)
    predictions_df = pd.read_csv(os.path.join(RESULTS_DIR, 'gtdb_predictions.csv'))

    if not os.path.exists(os.path.join(RESULTS_DIR, 'gtdb_stop_codons.csv')):
        get_stop_codons(predictions_df.gene_id.values)   
    if not os.path.exists(os.path.join(RESULTS_DIR, 'gtdb_copy_nums.csv')):
        get_copy_numbers()
    if not os.path.exists(os.path.join(RESULTS_DIR, 'gtdb_sec_trna_counts.csv')):
        get_sec_trna_counts(predictions.genome_id.unique())    


    copy_nums_df = pd.read_csv(os.path.join(RESULTS_DIR, 'gtdb_copy_nums.csv')) # , index_col=0)
    stop_codons_df = pd.read_csv(os.path.join(RESULTS_DIR, 'gtdb_stop_codons.csv')) # , index_col=0)
    sec_trna_counts_df = pd.read_csv(os.path.join(RESULTS_DIR, 'gtdb_sec_trna_counts.csv')) # , index_col=0)

    results_df = predictions_df.merge(copy_nums_df, how='left', left_on='genome_id', right_on='genome_id')
    results_df = results_df.merge(stop_codons_df, how='left', left_on='gene_id', right_on='gene_id')
    results_df = results_df.merge(sec_trna_counts_df, how='left', left_on='genome_id', right_on='genome_id')

    results_df = results_df.set_index('gene_id')
    results_df.to_csv(args.output_path)
    print(f'Results written to {args.output_path}')



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















