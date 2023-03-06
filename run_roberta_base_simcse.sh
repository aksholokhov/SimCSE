#!/bin/bash
#SBATCH --job-name="roberta_base"
#SBATCH --account=dynamicsai
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=1-23:00:00

#SBATCH --output=logs/slurm-%A_%a.out

NUM_GPU=4
PORT_ID=25228
export OMP_NUM_THREADS=8

source ~/.bashrc

module load gcc/11.2.0
module load cuda/11.8.0

conda activate compute

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path roberta-base \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/unsup_simcse_roberta_base \
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
