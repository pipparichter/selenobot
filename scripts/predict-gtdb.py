import os
import re
import tqdm 
from fabapi import * 
from files import EmbeddingsFile
from utils import WEIGHTS_DIR
import h5py

EMBEDDINGS_PATH = '/central/groups/fischergroup/prichter/gtdb/embeddings'
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'plm_model_weights.pth')

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


def get_copy_numbers(genome_id:str):
    '''Retrieve the copy numbers of genes related to selenoprotein synthesis for the genome.'''
    query = GetQuery('annotations_kegg')
    query.equal_to('genome_id', genome_id)
    query.equal_to('ko', [SELA_KO, SELB_KO, SELD_KO])

    page_df = query.next()
    df = []
    while page_df is not None:
        df.append(page_df)
        page_df = query.next()

    df = pd.concat(df)
    seld_copy_num = len(df.ko == SELD_KO)
    sela_copy_num = len(df.ko == SELA_KO)
    selb_copy_num = len(df.ko == SELB_KO)
    return seld_copy_num, sela_copy_num, selb_copy_num


def get_sec_trna_count(genome_id:str):
    '''Retrieve the number of selenocysteine tRNAs in the genome.'''
    query = GetQuery('metadata')
    query.equal_to('genome_id', genome_id) 

    df = query.submit()
    return df.sec_trna_count.values[0]


def get_stop_codon(gene_id:str):
    '''Retrieve the gene's stop codon.'''
    query = GetQuery('proteins')
    query.equal_to('gene_id', gene_id) 

    df = query.submit()
    return df.stop_codon.values[0]





if __name__ == '__main__':


    model = Classifier(latent_dim=1024, hidden_dim=512)
    model.load_state_dict(torch.load(WEIGHTS_PATH))


    df = pd.DataFrame({'id':dataset.ids, 'model_output':reporter.outputs, 'prediction':predictions})
    df['seq'] = dataset.seqs # Add sequences to the DataFrame. 
    df.set_index('id').to_csv(args.output)

    embedding_file_names = os.listdir(EMBEDDINGS_PATH) # Filenames are the RS_ or GB_ prefix followed by the genome ID. 
    genome_ids = [re.search('GC[AF]_\d{9}\.\d{1}', self.file_name).group(0) for file in embedding_files]

    for embedding_file_name, genome_id in zip(embedding_file, genome_ids):
        seld_copy_num, sela_copy_num, selb_copy_num = get_copy_numbers()
        sec_trna_count = get_sec_trna_count()
        
        embeddings_file = EmbeddingsFile(os.path.join(EMBEDDINGS_PATH, embedding_file_name))
        stop_codons = [get_stop_codon(gene_id) for gene_id in embedding_file_name.keys()]
            
        dataset = Dataset(embedding_file_name.dataframe()) # Instantiate a Dataset object with the embeddings. 
        reporter = model.predict(dataset)
        predictions = reporter.apply_threshold()




