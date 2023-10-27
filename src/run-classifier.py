from classifiers_jg import EmbeddingClassifier
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

classifier = EmbeddingClassifier()
classifier.load("emb_model.state_dict")


import pandas as pd
import h5py
import pickle
import numpy as np
import glob
import os
import Bio.SeqIO as SeqIO
from scipy.stats import fisher_exact

def getEmbedding(file_name):
    f = h5py.File(file_name)
    Xg = []
    gene_names = list(f.keys())
    for key in gene_names:
        Xg.append(f[key][()])
    f.close()
    Xg = pd.DataFrame(np.asmatrix(Xg),index=gene_names)
    return Xg

path_to_embeddings = "/groups/fischergroup/goldford/gtdb/embedding/"
path_to_annotations = "/groups/fischergroup/goldford/gtdb/ko_annotations/"
path_to_pfams = "/groups/fischergroup/goldford/gtdb/pfams/"

genomes = glob.glob("{fpath}*.h5".format(fpath=path_to_embeddings))
g = [x.split("/")[-1].split("_embedding.h5")[0] for x in genomes]
emebedding_file_df = pd.DataFrame({"gid":g,"embedding_file":genomes})

genomes = glob.glob("{apath}*.tab".format(apath=path_to_annotations))
g = [x.split("/")[-1].split(".ko.tab")[0].split("_protein")[0] for x in genomes]
annotation_df = pd.DataFrame({"gid":g,"annotation_file":genomes})

genomes = glob.glob("{apath}*.tsv".format(apath=path_to_pfams))
g = [x.split("/")[-1].split(".tsv")[0] for x in genomes]
pfam_df = pd.DataFrame({"gid":g,"pfam_file":genomes}).set_index("gid")

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
        "seld_copy_num":len(annots[annots.KO == "K01008"])
    }
    

# Convert the results dictionary to a DataFrame for easier viewing
results_df = pd.DataFrame(results_dict).T
results_df.to_pickle("selenoprotein_results.30Sep2023.pkl")




