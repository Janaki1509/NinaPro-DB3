#!/bin/bash
#SBATCH --job-name=minirocket_dryrun
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/%u/ninapro_bench/logs/%x_%A.out
#SBATCH --error=/home/%u/ninapro_bench/logs/%x_%A.err

set -euo pipefail

source ~/venvs/ninapro/bin/activate
cd ~/ninapro_bench

mkdir -p logs results/per_subject

TEST_SUBJECT="s1"

SCRATCH_DIR="${SCRATCH:-/scratch/$USER}/ninapro/${SLURM_JOB_ID}_dryrun"
mkdir -p "$SCRATCH_DIR"

python scripts/run_minirocket_loso.py \
  --base "<DB3_SUBJECTS_PATH_ON_CLIPPER>" \
  --test "$TEST_SUBJECT" \
  --out_json "results/per_subject/${TEST_SUBJECT}_dryrun.json" \
  --cpus "$SLURM_CPUS_PER_TASK" \
  --tmp "$SCRATCH_DIR" \
  --train_per_class 30 \
  --test_per_class 30 \
  --num_kernels 2000

rm -rf "$SCRATCH_DIR"
echo "Dry run complete."