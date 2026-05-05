#!/bin/bash
#SBATCH --job-name=minirocket_loso
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-11
#SBATCH --output=/home/%u/ninapro_bench/logs/%x_%A_%a.out
#SBATCH --error=/home/%u/ninapro_bench/logs/%x_%A_%a.err

set -euo pipefail

source ~/venvs/ninapro/bin/activate
cd ~/ninapro_bench

mkdir -p logs results/per_subject

TEST_SUBJECT="s${SLURM_ARRAY_TASK_ID}"

SCRATCH_DIR="${SCRATCH:-/scratch/$USER}/ninapro/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH_DIR"

python scripts/run_minirocket_loso.py \
  --base "<DB3_SUBJECTS_PATH_ON_CLIPPER>" \
  --test "$TEST_SUBJECT" \
  --out_json "results/per_subject/${TEST_SUBJECT}.json" \
  --cpus "$SLURM_CPUS_PER_TASK" \
  --tmp "$SCRATCH_DIR" \
  --train_per_class 120 \
  --test_per_class 80 \
  --num_kernels 10000

rm -rf "$SCRATCH_DIR"