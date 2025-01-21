#!/bin/bash
#SBATCH --job-name=embed_model_organisms
#SBATCH --time=10:00:00
#SBATCH --mem=100GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=END 
#SBATCH --main-user=prichter@caltech.edu

python embed.py --input-path ../data/model_organisms/gtdb_ecol_metadata.csv --feature-types plm --output-path ../data/model_organisms/gtdb_ecol.h5
python embed.py --input-path ../data/model_organisms/gtdb_bsub_metadata.csv --feature-types plm --output-path ../data/model_organisms/gtdb_bsub.h5
python embed.py --input-path ../data/model_organisms/gtdb_mtub_metadata.csv --feature-types plm --output-path ../data/model_organisms/gtdb_mtub.h5
python embed.py --input-path ../data/model_organisms/gtdb_paer_metadata.csv --feature-types plm --output-path ../data/model_organisms/gtdb_paer.h5
python embed.py --input-path ../data/model_organisms/gtdb_afis_metadata.csv --feature-types plm --output-path ../data/model_organisms/gtdb_afis.h5