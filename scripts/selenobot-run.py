'''A script for running a trained classifier on embedded genome data. This script should be run on the HPC, which is where the 
embedding and annotation data is stored.'''

from selenobot.classifiers import Classifier
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import h5py
import pickle
import numpy as np
import glob # For file pattern matching.
import os
import Bio.SeqIO as SeqIO
from scipy.stats import fisher_exact
from datetime import date

classifier = EmbeddingClassifier()
classifier.load("emb_model.state_dict")

DATE = date.today().strftime('%d.%m.%y')
ANNOTATION_DIR = '/groups/fischergroup/goldford/gtdb/ko_annotations/' # Path to KO annotations on HPC.
EMBEDDING_DIR = '/groups/fischergroup/goldford/gtdb/embedding/' # Path to embeddings on HPC.

def load_embedding(path:str):
    '''Load PLM embeddings from a file at the specified path. Each file contains embedded sequences for a
    single genome, stored as an HDF file.'''
    f = h5py.File(file_name)
    df = []
    gene_ids = list(f.keys())
    for key in gene_ids:
        data.append(f[key][()]) # What is this doing?
    f.close()
    df = pd.DataFrame(np.asmatrix(df), index=gene_ids)
    return df


if __name__ == '__main__':

    embedding_files = glob.glob(f'{EMBEDDING_DIR}*.h5') # Does this list the entire path?
    genome_ids = [f.split('/')[-1].replace('_embedding.h5', '') for f in embedding_files]
    # Map genome IDs to the corresponding embedding file. 
    embedding_files = {g:f for f, g in zip(embedding_files, genome_ids)}
    
    annotation_files = glob.glob(f'{EMBEDDING_DIR}*.tab') # Does this list the entire path?
    genome_ids = [f.split('/')[-1].replace('_protein.ko.tab', '') for f in annotation_files]
    # Map genome IDs to the corresponding embedding file. 
    annotation_files = {g:f for f, g in zip(annotation_files, genome_ids)}



df_embedding_annotations = annotation_df.set_index("gid").join(emebedding_file_df.set_index("gid"))
df_embedding_annotations = df_embedding_annotations.dropna().join(pfam_df)


# Initialize a dictionary to store the results

results_dict = {}
for genome, row in df_embedding_annotations.iterrows():
    EmbeddingMatrix = getEmbedding(row.embedding_file)
    #break
    labels = classifier(torch.tensor(EmbeddingMatrix.values)).numpy().T[0]
    
    results = pd.DataFrame({"gene": EmbeddingMatrix.index,  "genome": genome, "selenoprotein": labels})
    results["gene"] = results['gene'].apply(lambda x: x.split(" ")[0])
    annots = pd.read_csv(row.annotation_file, sep="\t", skiprows=[1]).dropna()
    annots["gene name"] = annots["gene name"].apply(lambda x: x.replace(".", "_"))
    annots = annots.set_index("gene name")
    hits = results[results.selenoprotein > 0.5]
    hits_with_annotation = hits.set_index("gene").join(annots)
    ko_map = hits_with_annotation.dropna().groupby("KO").count()["genome"].to_dict()

    
    # Record the required details
    results_dict[genome] = {
        "total_hits": len(hits),
        "hits_with_annotation": len(hits_with_annotation.dropna()),
        "total_genes": EmbeddingMatrix.shape[0],
        "total_genes_with_annotation": len(annots),
        "hits_with_annotation": ko_map,
        "selD_copy_num":len(annots[annots.KO == "K01008"])
    }
    

# Convert the results dictionary to a DataFrame for easier viewing
results_df = pd.DataFrame(results_dict).T
results_df.to_pickle("selenoprotein_results.30Sep2023.pkl")




