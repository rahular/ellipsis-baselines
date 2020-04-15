#!/bin/bash

#SBATCH --job-name=drqa-evaluate
#SBATCH --ntasks=6
#SBATCH --output=logs/predictions.out
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanx:1

source activate drqa
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

run()
{
    # $1 => DATA and $2 => MODEL
    python scripts/reader/predict.py \
        ../data/$1/test.json \
        --model=./models/$2.mdl \
        --embedding-file=./embeddings/glove.840B.300d.txt \
        --out-dir=./models \
        --tokenizer=spacy \
        --gpu=0 \
        --top-n=1 \
        --official
    mv ./models/test-$2.preds ./models/$2ON$1.preds
    echo "Evaluating $2 model on $1 data..."
    python ../evaluate-v1.1.py ../data/$1/test.json ./models/$2ON$1.preds

}

# predict on sluice test set using the sluice+vpe model
run sluice sluice_vpe
