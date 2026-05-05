#!/usr/bin/env python3
"""
run_agentic_search.py
======================
Agentic hyperparameter search — Professor Item 5.

Run this on the HPC LOGIN NODE only.
It generates and submits Slurm jobs automatically,
then collects results and prints the best configuration.

What it searches:
  - latent_dim:     [64, 128, 256]
  - lr:             [1e-4, 5e-4, 1e-3]
  - dropout:        [0.2, 0.3, 0.4]
  - alpha (adv):    [0.3, 0.5, 0.8]

Each combination = one Slurm job on subject s1 (fast proxy).
Best config then gets submitted as full 11-subject array.

Usage (on HPC login node ONLY):
  python ~/ninapro_db3/scripts/run_agentic_search.py \
      --model adversarial \
      --out_dir ~/ninapro_db3/runs/search \
      --results_csv ~/ninapro_db3/results/search_results.csv
"""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from itertools import product

# Safety: only run on login node (this script just submits jobs)
import socket
hostname = socket.gethostname()
if not hostname.startswith("port") and "login" not in hostname:
    print(f"WARNING: hostname is {hostname}. "
          f"This script should run on the login node to submit jobs.")


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=search_{job_id}
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output={log_dir}/search_{job_id}_%j.out
#SBATCH --error={log_dir}/search_{job_id}_%j.err

source ~/venvs/ninapro/bin/activate

python {script_path} \\
    --base /mnt/home/chandraj/ninapro_db3/subjects \\
    --test s1 \\
    --out_json {out_json} \\
    --epochs 30 \\
    --latent_dim {latent_dim} \\
    --lr {lr} \\
    --dropout {dropout} \\
    {extra_args} \\
    --cpus 4

echo "Job {job_id} done"
"""


def submit_job(sbatch_content, sbatch_path):
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)
    os.chmod(sbatch_path, 0o755)
    result = subprocess.run(
        ['sbatch', sbatch_path],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR submitting {sbatch_path}: {result.stderr}")
        return None
    job_id = result.stdout.strip().split()[-1]
    return job_id


def wait_for_jobs(job_ids, poll_interval=30, timeout=3600):
    """Poll squeue until all jobs finish or timeout."""
    print(f"  Waiting for {len(job_ids)} jobs to finish...")
    start = time.time()
    pending = set(job_ids)
    while pending and (time.time() - start) < timeout:
        result = subprocess.run(
            ['squeue', '--jobs', ','.join(pending), '--noheader'],
            capture_output=True, text=True)
        still_running = set()
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                jid = line.split()[0]
                still_running.add(jid)
        finished = pending - still_running
        if finished:
            print(f"  Finished: {sorted(finished)}")
        pending = still_running
        if pending:
            print(f"  Still running: {len(pending)} jobs... "
                  f"(elapsed: {int(time.time()-start)}s)")
            time.sleep(poll_interval)
    if pending:
        print(f"  TIMEOUT: {len(pending)} jobs still running")
    else:
        print("  All jobs finished.")


def collect_results(result_files):
    results = []
    for config, json_path in result_files:
        if not os.path.isfile(json_path):
            print(f"  Missing result: {json_path}")
            continue
        with open(json_path) as f:
            data = json.load(f)
        results.append({**config, 'balanced_acc': data.get('balanced_acc', 0)})
    return sorted(results, key=lambda x: -x['balanced_acc'])


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',   default='adversarial',
                    choices=['adversarial', 'latent', 'conditional'])
    ap.add_argument('--out_dir', default=os.path.expanduser(
                    '~/ninapro_db3/runs/search'))
    ap.add_argument('--results_csv', default=os.path.expanduser(
                    '~/ninapro_db3/results/search_results.csv'))
    ap.add_argument('--submit_full', action='store_true',
                    help='After search, submit best config as full 11-subject array')
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.expanduser('~/ninapro_db3/logs')
    os.makedirs(log_dir, exist_ok=True)

    proj = os.path.expanduser('~/ninapro_db3')

    # Script paths
    script_map = {
        'adversarial': f'{proj}/scripts/run_adversarial_latent_loso.py',
        'latent':      f'{proj}/scripts/run_latent_loso.py',
        'conditional': f'{proj}/scripts/run_conditional_latent_loso.py',
    }
    script_path = script_map[args.model]

    # Search space
    latent_dims = [64, 128, 256]
    lrs         = [1e-4, 5e-4, 1e-3]
    dropouts    = [0.2, 0.3, 0.4]
    alphas      = [0.3, 0.5, 0.8] if args.model == 'adversarial' else [None]

    configs = []
    for ld, lr, dr, alpha in product(latent_dims, lrs, dropouts, alphas):
        configs.append({
            'latent_dim': ld,
            'lr':         lr,
            'dropout':    dr,
            'alpha':      alpha,
        })

    print(f"Agentic search: {len(configs)} configurations")
    print(f"Model: {args.model}")
    print(f"Output: {args.out_dir}")
    print()

    # Submit all jobs
    submitted = []
    result_files = []
    for i, cfg in enumerate(configs):
        job_id  = f"{args.model}_{i:03d}"
        out_json = os.path.join(args.out_dir, f'{job_id}.json')
        extra = ''
        if cfg['alpha'] is not None:
            extra = f'--alpha {cfg["alpha"]}'

        sbatch_content = SBATCH_TEMPLATE.format(
            job_id=job_id,
            log_dir=log_dir,
            script_path=script_path,
            out_json=out_json,
            latent_dim=cfg['latent_dim'],
            lr=cfg['lr'],
            dropout=cfg['dropout'],
            extra_args=extra,
        )
        sbatch_path = os.path.join(args.out_dir, f'{job_id}.sbatch')
        slurm_id = submit_job(sbatch_content, sbatch_path)
        if slurm_id:
            submitted.append(slurm_id)
            result_files.append((cfg, out_json))
            print(f"  Submitted job {job_id} -> Slurm {slurm_id} "
                  f"(ld={cfg['latent_dim']} lr={cfg['lr']} "
                  f"dr={cfg['dropout']} alpha={cfg['alpha']})")

    print(f"\nSubmitted {len(submitted)} jobs. Waiting for results...")
    print("(You can also Ctrl+C and check later with --collect_only)")

    # Wait for all to finish
    wait_for_jobs(submitted, poll_interval=60, timeout=7200)

    # Collect and rank results
    results = collect_results(result_files)

    print("\n" + "="*60)
    print("SEARCH RESULTS (ranked by LOSO balanced accuracy on S1)")
    print("="*60)
    for rank, r in enumerate(results[:10], 1):
        print(f"  #{rank:2d}  bacc={r['balanced_acc']:.4f}  "
              f"ld={r['latent_dim']}  lr={r['lr']}  "
              f"dr={r['dropout']}  alpha={r.get('alpha','—')}")

    # Save CSV
    if results:
        keys = list(results[0].keys())
        with open(args.results_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(results)
        print(f"\nFull results saved to: {args.results_csv}")

    # Report best
    if results:
        best = results[0]
        print(f"\nBest configuration:")
        print(f"  balanced_acc = {best['balanced_acc']:.4f}")
        print(f"  latent_dim   = {best['latent_dim']}")
        print(f"  lr           = {best['lr']}")
        print(f"  dropout      = {best['dropout']}")
        if best.get('alpha'):
            print(f"  alpha        = {best['alpha']}")

        if args.submit_full:
            print("\nSubmitting best config as full 11-subject array...")
            _submit_full_array(best, args.model, script_path,
                               log_dir, args.out_dir, proj)


def _submit_full_array(best, model_name, script_path, log_dir, out_dir, proj):
    """Submit the best config as an 11-subject LOSO array."""
    extra = f'--alpha {best["alpha"]}' if best.get('alpha') else ''
    template = f"""#!/bin/bash
#SBATCH --job-name=best_{model_name}
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --array=1-11
#SBATCH --output={log_dir}/best_{model_name}_s%a_%j.out
#SBATCH --error={log_dir}/best_{model_name}_s%a_%j.err

source ~/venvs/ninapro/bin/activate
S=$SLURM_ARRAY_TASK_ID

python {script_path} \\
    --base /mnt/home/chandraj/ninapro_db3/subjects \\
    --test s${{S}} \\
    --out_json {out_dir}/best_s${{S}}.json \\
    --latent_npz {out_dir}/best_s${{S}}_latents.npz \\
    --epochs 50 \\
    --latent_dim {best['latent_dim']} \\
    --lr {best['lr']} \\
    --dropout {best['dropout']} \\
    {extra} \\
    --cpus 4
"""
    sbatch_path = os.path.join(out_dir, f'best_{model_name}_array.sbatch')
    with open(sbatch_path, 'w') as f:
        f.write(template)
    result = subprocess.run(['sbatch', sbatch_path],
                            capture_output=True, text=True)
    print(f"Full array submitted: {result.stdout.strip()}")


if __name__ == '__main__':
    main()
