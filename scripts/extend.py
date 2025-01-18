from selenobot.genes import Genome, Gene
import argparse
import pandas as pd
import os
from tqdm import tqdm
from selenobot.utils import default_output_path


# The information I need to extend the sequence is... 
# 1. The scaffold ID, which is (I think) the first part of the gene ID. 
# 2. Whether or not the gene runs off the contig. 

# NOTE: Are the start and stop positions zero-indexed? No, the start position is one-indexed. 
# NOTE: Do the start and stop positions refer to the contig, or entire genome? 
# NOTE: Does the "partial" flag always mean the gene runs off the boundary? Or can there be other reasons?

required_cols = ['start', 'stop', 'genome_id', 'strand', 'id', 'partial', 'ncbi_translation_table']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', default=None, type=str)
    parser.add_argument('--output-path', default=None, type=str)
    parser.add_argument('--genomes-dir', default='../data/gtdb_subset_genomes')
    parser.add_argument('--output-format', choices=['faa', 'fna', 'csv'], default='csv')
    parser.add_argument('--genome-file-name-format', default='{genome_id}_genomic.fna')
    args = parser.parse_args()

    if output_path is None:
        output_path = default_output_path(input_path, operation='extend', extension=args.output_formmat)
    else:
        output_path = args.output_path
    
    df = pd.read_csv(args.input_path, index_col=0, dtype={'partial':str}, usecols=required_cols)
    partial_filter = ~df.partial.str.match('00')
    print(f'Removing {partial_filter.sum()} partial sequences.')
    df = df[~partial_filter]

    extended_df = [] # This will store the extended sequences.
    pbar = tqdm(total=len(df), desc=f'Extending sequences in {args.input_path}') 
    n_failures = 0 

    for genome_id, genome_df in df.groupby('genome_id'):
        genome_file_name = args.genome_file_name_format.format(genome_id=genome_id)
        genome_file_path = os.path.join(args.genomes_dir, genome_file_name)
        if not os.path.exists(genome_file_path):
            print(f'Could not find genome file {genome_file_path}. Skipping extensions for that genome.')
            pbar.update(len(genome_df))
            continue
        genome = Genome(genome_file_path)

        for row in genome_df.itertuples():
            gene = Gene(row.Index, genome, start=row.start - 1, stop=row.stop, strand=row.strand)
            gene.check() # Make sure the gene has valid start and stop codons. 

            extended_gene = gene.extend(error='ignore')
            extended_row = extended_gene.info(translation_table=row.ncbi_translation_table)
            extended_row['genome_id'] = genome_id # Add the genome ID to the information in the new DataFrame. 
            extended_df.append(extended_row)

            if extended_gene.extension_size == 0:
                n_failures += 1
            failure_rate = 100 * (n_failures / len(extended_df))
            pbar.update(1)
            pbar.set_description(f'Extending sequences in {args.input_path}. Failure rate {failure_rate:.2f}%')

    
    extended_df = pd.DataFrame(extended_df).set_index('gene_id')
    extended_df.index.name = 'id'
    extended_df.to_csv(output_path)
    print(f'Extended sequences written to {output_path}')








