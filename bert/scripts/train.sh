#!/bin/bash

#SBATCH --job-name=bert-sluice
#SBATCH --ntasks=1
#SBATCH --output=logs/sluice.out
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanx:1
#SBATCH --mem=60GB

source activate transformers
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

DATA_DIR=../data/sluice
DEV_DIR=../data/sluice

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $DATA_DIR/train.json \
  --predict_file $DEV_DIR/dev.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./models/sluice \
  --overwrite_output_dir