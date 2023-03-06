#!/bin/bash
#SBATCH --job-name="bert_large"
#SBATCH --account=dynamicsai
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=23:00:00

#SBATCH --output=logs/slurm-%A_%a.out

source ~/.bashrc

module load gcc/11.2.0
module load cuda/11.8.0

conda activate compute

python train.py \
    --model_name_or_path bert-large-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/unsup_simcse_bert_large \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --save_steps 10 \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_mlm \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
