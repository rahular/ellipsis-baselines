#!/bin/bash

#SBATCH --job-name=qanet-predict
#SBATCH --ntasks=1
#SBATCH --output=logs/predictions.out
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanx:1

source activate allennlp
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

run()
{
    # $1 => DATA and $2 => MODEL
    allennlp predict ./models/$2/model.tar.gz ../data/$1/test.json --output-file=./models/$2/$1.preds --include-package=my_library --cuda-device=0 --predictor=qanet_predictor --use-dataset-reader
}

# predict on sluice test set using the sluice+vpe model
run sluice sluice_vpe