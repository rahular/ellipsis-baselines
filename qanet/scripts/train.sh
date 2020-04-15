#!/bin/bash

#SBATCH --job-name=qanet-sluice
#SBATCH --ntasks=1
#SBATCH --output=logs/sluice.out
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanx:1
#SBATCH --mem=60GB

source activate allennlp
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

allennlp train qanet_sluice.json --serialization-dir=./models/sluice --include-package=my_library