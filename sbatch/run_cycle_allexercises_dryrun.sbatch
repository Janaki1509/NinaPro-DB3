#!/bin/bash
#SBATCH --job-name=cyc_all_dry
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/chandraj/ninapro_db3/logs/cyc_all_dry_%j.out
#SBATCH --error=/mnt/home/chandraj/ninapro_db3/logs/cyc_all_dry_%j.err

if [[ "$HOSTNAME" == port* ]]; then echo "ERROR: login node"; exit 1; fi

echo "=== CYCLE ALL EXERCISES DRY RUN s1 | $HOSTNAME | $(date) ==="
source ~/venvs/ninapro/bin/activate

PROJ=$HOME/ninapro_db3
OUT=$PROJ/runs/cycle_all
mkdir -p $OUT

python $PROJ/scripts/run_cycle_allexercises_loso.py \
    --base /mnt/home/chandraj/ninapro_db3/subjects \
    --test s1 \
    --out_json   $OUT/s1_cycle_all.json \
    --latent_npz $OUT/s1_latents.npz \
    --epochs 20 \
    --lambda_recon 1.0 \
    --lambda_cycle_max 1.0 \
    --cpus 4

echo "=== Done. Check: $OUT/s1_cycle_all.json ==="
