#!/bin/bash
#SBATCH --job-name=mlp_loso
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-11
#SBATCH --output=/mnt/home/%u/ninapro_db3/logs/%x_%A_%a.out
#SBATCH --error=/mnt/home/%u/ninapro_db3/logs/%x_%A_%a.err

set -euo pipefail

source ~/venvs/ninapro/bin/activate
cd ~/ninapro_db3

mkdir -p logs results/per_subject runs

TEST_SUBJECT="s${SLURM_ARRAY_TASK_ID}"

python scripts/run_mlp_loso.py \
  --base "/mnt/home/chandraj/ninapro_db3/subjects" \
  --test "$TEST_SUBJECT" \
  --out_json "results/per_subject/${TEST_SUBJECT}_mlp.json" \
  --cpus "$SLURM_CPUS_PER_TASK" \
  --train_per_class 120 \
  --test_per_class 80 \
  --epochs 20 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --hidden_dim 128 \
  --dropout 0.2
