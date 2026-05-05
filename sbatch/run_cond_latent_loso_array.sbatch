#!/bin/bash
#SBATCH --job-name=cond_loso
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=1-11
#SBATCH --output=/mnt/home/chandraj/ninapro_db3/logs/cond_s%a_%j.out
#SBATCH --error=/mnt/home/chandraj/ninapro_db3/logs/cond_s%a_%j.err

if [[ "$HOSTNAME" == port* ]]; then
    echo "ERROR: on login node."; exit 1
fi

S=$SLURM_ARRAY_TASK_ID
echo "=== CONDITIONAL LATENT s${S} | $HOSTNAME | $(date) ==="
source ~/venvs/ninapro/bin/activate

PROJ=$HOME/ninapro_db3
OUT=$PROJ/runs/cond_latent
mkdir -p $OUT

python $PROJ/scripts/run_conditional_latent_loso.py \
    --base /mnt/home/chandraj/ninapro_db3/subjects \
    --test s${S} \
    --out_json   $OUT/s${S}_cond_latent.json \
    --latent_npz $OUT/s${S}_latents.npz \
    --epochs 50 \
    --hidden_dim 256 \
    --cpus 4

echo "=== s${S} done ==="
