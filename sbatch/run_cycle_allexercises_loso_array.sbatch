#!/bin/bash
#SBATCH --job-name=cyc_all
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=1-11
#SBATCH --output=/mnt/home/chandraj/ninapro_db3/logs/cyc_all_s%a_%j.out
#SBATCH --error=/mnt/home/chandraj/ninapro_db3/logs/cyc_all_s%a_%j.err

if [[ "$HOSTNAME" == port* ]]; then echo "ERROR: login node"; exit 1; fi

S=$SLURM_ARRAY_TASK_ID
echo "=== CYCLE ALL EXERCISES s${S} | $HOSTNAME | $(date) ==="
source ~/venvs/ninapro/bin/activate

PROJ=$HOME/ninapro_db3
OUT=$PROJ/runs/cycle_all
mkdir -p $OUT

python $PROJ/scripts/run_cycle_allexercises_loso.py \
    --base /mnt/home/chandraj/ninapro_db3/subjects \
    --test s${S} \
    --out_json   $OUT/s${S}_cycle_all.json \
    --latent_npz $OUT/s${S}_latents.npz \
    --epochs 100 \
    --lambda_recon 1.0 \
    --lambda_cycle_max 1.0 \
    --cpus 4

echo "=== s${S} done ==="
