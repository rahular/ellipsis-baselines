#!/bin/bash

#SBATCH --job-name=drqa-prepare
#SBATCH --output=logs/prepare.out
#SBATCH --time=1:00:00
#SBATCH --partition=image1

source activate drqa
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

# preprocess sluice data before training
python scripts/reader/preprocess.py --tokenizer=spacy ../data/sluice ./processed-data/sluice --split train
python scripts/reader/preprocess.py --tokenizer=spacy ../data/sluice ./processed-data/sluice --split dev
python scripts/reader/preprocess.py --tokenizer=spacy ../data/sluice ./processed-data/sluice --split test