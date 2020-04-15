#!/bin/bash

#SBATCH --job-name=drqa-sluice
#SBATCH --ntasks=6
#SBATCH --output=logs/sluice.out
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanx:1

source activate drqa
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

python scripts/reader/train.py \
    --gpu=0 \
    --random-seed=42 \
    --num-epochs=20 \
    --batch-size=32 \
    --test-batch-size=32 \
    --data-workers=5 \
    --model-dir=./models \
    --model-name=sluice \
    --data-dir=./processed-data/sluice \
    --train-file=train-processed-spacy.txt \
    --dev-file=dev-processed-spacy.txt \
    --dev-json=dev.json \
    --embed-dir=./embeddings \
    --embedding-file=glove.840B.300d.txt \
    --valid-metric=f1 \
    --display-iter=100 \
    --uncased-question=true \
    --uncased-doc=true \
    --parallel=true \
    --checkpoint=true \
    --expand-dictionary=true \
    --official-eval=true