#!/bin/bash
#SBATCH --job-name=cond_dry
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/chandraj/ninapro_db3/logs/cond_dry_%j.out
#SBATCH --error=/mnt/home/chandraj/ninapro_db3/logs/cond_dry_%j.err

if [[ "$HOSTNAME" == port* ]]; then
    echo "ERROR: on login node. Use sbatch only."; exit 1
fi

echo "=== CONDITIONAL LATENT DRY RUN s1 | $HOSTNAME | $(date) ==="
source ~/venvs/ninapro/bin/activate

PROJ=$HOME/ninapro_db3
OUT=$PROJ/runs/cond_latent
mkdir -p $OUT

python $PROJ/scripts/run_conditional_latent_loso.py \
    --base /mnt/home/chandraj/ninapro_db3/subjects \
    --test s1 \
    --out_json  $OUT/s1_cond_latent.json \
    --latent_npz $OUT/s1_latents.npz \
    --epochs 20 \
    --cpus 4

echo "=== Done. Check: $OUT/s1_cond_latent.json ==="
