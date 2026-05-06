#!/bin/bash
#SBATCH --job-name=mlp_dryrun
#SBATCH --partition=debug
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/mnt/home/%u/ninapro_db3/logs/%x_%A.out
#SBATCH --error=/mnt/home/%u/ninapro_db3/logs/%x_%A.err

set -euo pipefail

source ~/venvs/ninapro/bin/activate
cd ~/ninapro_db3

mkdir -p logs results/per_subject runs

TEST_SUBJECT="s1"

python scripts/run_mlp_loso.py \
  --base "/mnt/home/chandraj/ninapro_db3/subjects" \
  --test "$TEST_SUBJECT" \
  --out_json "results/per_subject/${TEST_SUBJECT}_mlp_dryrun.json" \
  --cpus "$SLURM_CPUS_PER_TASK" \
  --train_per_class 30 \
  --test_per_class 30 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --hidden_dim 128 \
  --dropout 0.2
