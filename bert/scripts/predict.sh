#!/bin/bash

#SBATCH --job-name=bert-eval
#SBATCH --ntasks=1
#SBATCH --output=logs/bert-eval.out
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB

source activate transformers
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

run()
{
  # $1 => DATA and $2 => MODEL
  DATA_DIR=../data/$1

  python run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-cased \
    --do_eval \
    --do_lower_case \
    --train_file $DATA_DIR/train.json \
    --predict_file $DATA_DIR/test.json \
    --per_gpu_train_batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./models/$2 \
    --overwrite_output_dir

  mv ./models/$2/predictions_.json ./models/$2/predictions_$1.json
  mv ./models/$2/nbest_predictions_.json ./models/$2/nbest_predictions_$1.json
  echo "Evaluating $2 model on $1 data..."
  python ../evaluate-v1.1.py $DATA_DIR/test.json ./models/$2/predictions_$1.json
}

# predict on sluice test set using the sluice+vpe model
run sluice sluice_vpe


