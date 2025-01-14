

from selenobot.files import FASTAFile
import pandas as pd
import numpy as np 
import os 
import re 
import h5py
from tqdm import tqdm
import argparse

class EmbeddingsFile():
    '''Handles reading files containing Prot-T5 embeddings, which are stored on HPC as HDF files'''

    def __init__(self, path:str):
        
        file = h5py.File(path, 'r') # For whatever reason, using pandas HDFStore doesn't work. 
        self.unique_id_to_key_map = {re.search('ID=([^;]+)', key).group(1):key for key in file.keys()} # Map the unique IDs to the HDF file key. 
        self.unique_ids = sorted(list(unique_id_to_key_map.keys())) # Order the IDs so I can easily match them with metadata later. 
        # Josh modified the gene IDs in a way which makes them annoying to match with the gene metadata I have. It is easier to extract the
        # Prodigal-assigned unique identifiers (ID={unique_id}), which are of the form {ith_contig}_{nth_gene_on_contig}. 
        
        embeddings = []
        for unique_id in self.unique_ids:
            key = self.unique_id_to_key_map[unique_id]
            emb = np.empty(file[key].shape, dtype=np.float32)
            data[key].read_direct(emb) # read_direct avoids the copying step involved when accessing the dataset with slices. 
            embeddings.append(np.expand_dims(emb, axis=0)) # Expand dimensions so concatenation works correctly. 
        self.embeddings = np.concatenate(embeddings, axis=0)

        file.close()
        
    def to_df(self):
        df = pd.DataFrame(self.embeddings, index=self.unique_ids)
        df.index.name = 'ID' 
        return df


def load_proteins(genome_id:str, prefix:str, dir_:str=None, file_name_format:str='{prefix}_{genome_id}_protein.faa') -> pd.DataFrame:

    file_name = file_name_format.format(genome_id=genome_id, prefix=prefix)
    path = os.path.join(dir_, file_name)

    def parse_description(description:str):
        pattern = r'# ([\d]+) # ([\d]+) # ([-1]+) # ID=([^;]+);partial=([^;]+);start_type=([^;]+);rbs_motif=([^;]+);rbs_spacer=([^;]+);gc_cont=([\.\w]+)'
        columns = ['start', 'stop', 'strand', 'ID', 'partial', 'start_type', 'rbs_motif', 'rbs_spacer', 'gc_content']
        match = re.search(pattern, description)
        return {col:match.group(i + 1) for i, col in enumerate(columns)}

    df = FASTAFile(f'../data/gtdb_subset_proteins/{file_name}').to_df(parse_description=False)
    df = pd.concat([df, pd.DataFrame([parse_description(d) for d in df.description], index=df.index)], axis=1)
    df['prefix'] = prefix
    df['genome_id'] = genome_id
    return df


def load_embeddings(genome_id:str, prefix:str, dir_:str=None, file_name_format:str='{prefix}_{genome_id}_protein.faa') -> pd.DataFrame:

    file_name = file_name_format.format(genome_id=genome_id, prefix=prefix)
    path = os.path.join(dir_, file_name)
    file = EmbeddingsFile(path)
    return file.to_df()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genome-metadata-path', default='../data/gtdb_bacteria_genome_metadata.csv')
    parser.add_argument('--embeddings-dir', default='/central/groups/fischergroup/goldford/gtdb/embedding/')
    parser.add_argument('--proteins-dir', default='/central/groups/fischergroup/prichter/selenobot/data/gtdb_proteins/')
    parser.add_argument('--proteins-file-name-format', default='{prefix}_{genome_id}_protein.faa')
    parser.add_argument('--embeddings-file-name-format', default='{prefix}_{genome_id}_embedding.h5')
    parser.add_argument('--output-dir', default='/central/groups/fischergroup/prichter/selenobot/data/gtdb_embeddings/')

    args = parser.parse_args()

    genome_metadata_df = pd.read_csv(args.genome_metadata_path, index_col=0)
    genome_metadata_columns = genome_metadata_df.columns

    pbar = tqdm(total=len(genome_metadata_df), desc='Reading genome data...')
    for row in genome_metadata_df.itertuples():
        genome_id, prefix = row.Index, row.prefix
        proteins_df = load_proteins(genome_id, prefix, dir_=args.proteins_dir, file_name_format=args.proteins_file_name_format)
        embeddings_df = load_embeddings(genome_id, prefix, dir_=args.embeddings_dir, file_name_format=args.embeddings_file_name_format)

        # Remove sequences which exceed the maximum length specification. 
        length_filter = proteins_df.seq.apply(len) < 2000
        embeddings_df = embeddings_df[length_filter]
        proteins_df = proteins_df[length_filter]

        assert np.all(np.equal(embeddings_df.index, proteins_df.ID)), 'The IDs in the proteins and embeddings DataFrames do not match.'
        embeddings_df.index = proteins_df.index 
        embeddings_df.index.name = 'id'

        # Create a metadata DataFrame with the genome and protein metadata. 
        metadata_df = proteins_df.copy()
        for col in genome_metadata_columns:
            metadata_df[col] = getattr(row, col)

        store = pd.HDFStore(os.path.join(args.output_dir, f'{genome_id}.h5'), 'w')
        store.put('metadata', metadata_df)
        store.put('plm', embeddings_df)
        store.close()

        pbar.update(1)
    



