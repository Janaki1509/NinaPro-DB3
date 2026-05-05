#!/bin/bash
#SBATCH --job-name=latent_dry
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/chandraj/ninapro_db3/logs/latent_dry_%j.out
#SBATCH --error=/mnt/home/chandraj/ninapro_db3/logs/latent_dry_%j.err

# Safety check
if [[ "$HOSTNAME" == port* ]]; then
    echo "ERROR: on login node. Use sbatch only."; exit 1
fi

echo "=== LATENT DRY RUN (s1 only) | Host: $HOSTNAME | $(date) ==="

source ~/venvs/ninapro/bin/activate

PROJ=$HOME/ninapro_db3
OUT=$PROJ/runs/latent
mkdir -p $OUT $PROJ/logs

python $PROJ/scripts/run_latent_loso.py \
    --base /mnt/home/chandraj/ninapro_db3/subjects \
    --test s1 \
    --out_json  $OUT/s1_latent.json \
    --latent_npz $OUT/s1_latents.npz \
    --epochs 20 \
    --cpus 4

echo "=== Done. Check: $OUT/s1_latent.json ==="
