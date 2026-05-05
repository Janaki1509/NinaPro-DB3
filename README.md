# NinaPro DB3: Reproducible Cross-Subject EMG Gesture Classification

Reproducible benchmark for cross-subject surface EMG gesture recognition on NinaPro DB3, with leave-one-subject-out evaluation, domain shift analysis, confidence calibration, and comparison of multiple baseline and neural models.

## Project Overview

This project studies how well EMG gesture-recognition models generalize to unseen amputee subjects.  
Instead of relying on within-subject performance, the benchmark uses leave-one-subject-out (LOSO) evaluation to simulate a realistic clinical setting.

The capstone compares several methods, including:
- Majority baseline
- MiniROCKET
- MLP
- 1D CNN
- Conditional Latent CNN
- Adversarial Latent CNN
- Cycle-consistent autoencoder

It also analyzes subject-to-subject domain shift using Maximum Mean Discrepancy (MMD) and evaluates confidence calibration with temperature scaling.

## Why This Matters

Published EMG results often look strong under within-subject testing, but performance drops sharply when the model must predict for a new subject.  
That gap matters for prosthetic control, where a system must generalize to a new patient rather than memorize one person’s signal pattern.

## Dataset

- **Dataset:** NinaPro DB3
- **Subjects:** 11 transradial amputees
- **Signals:** 12 sEMG channels
- **Primary task:** Exercise 1, 17 gesture classes
- **Preprocessing:** 150 ms windows, 50% overlap, REST removed, channel-wise normalization

## Repo Structure

- `scripts/` — Python scripts for training, evaluation, plotting, and result processing
- `sbatch/` — Slurm job scripts used on Clipper HPC
- `results/` — CSV summaries and per-subject outputs
- `docs/` — capstone paper, poster, and summary files
- `figures/` — final figures for the report and LinkedIn
- `data/README.md` — instructions for obtaining NinaPro DB3 locally

## Methods

Models evaluated in this project:
- Majority class baseline
- MiniROCKET
- MLP
- Latent CNN
- Conditional Latent CNN
- Adversarial Latent CNN
- Cycle-consistent cross-subject autoencoder

Analysis methods:
- LOSO balanced accuracy
- Controlled ablation
- MMD domain shift analysis
- Confidence calibration with temperature scaling
- t-SNE latent space visualization
- Agentic hyperparameter search over multiple configurations

## Key Findings

- Evaluation protocol explains most of the apparent gap to published within-subject results.
- Cross-subject EMG classification is much harder than within-subject testing suggests.
- Higher inter-subject MMD is associated with lower LOSO performance.
- The cycle-consistent autoencoder achieved the best LOSO balanced accuracy in this study.
- Temperature scaling improved calibration and reduced overconfident predictions.

## Reproducibility

This repository is designed to be reproducible on Clipper HPC or a similar environment.

Typical workflow:
1. Install dependencies.
2. Place NinaPro DB3 files locally.
3. Run preprocessing / fold generation scripts.
4. Submit Slurm jobs from `sbatch/`.
5. Merge per-subject outputs into summary CSVs.
6. Generate figures from the saved results.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements_hpc.txt
```

If you rename it to `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

## Running the Project

Example commands depend on the model and job script.  
Common entry points include:
- `scripts/run_mlp_loso.py`
- `scripts/run_latent_loso.py`
- `scripts/run_cycle_autoencoder_loso.py`
- `scripts/run_minirocket_loso.py`

For HPC, use the corresponding files in `sbatch/`.

## Results

Final summary tables are stored in `results/` and can be regenerated using the plotting and merge scripts in `scripts/`.

## Data Access

The raw NinaPro DB3 dataset is not included in this repository.  
To reproduce the experiments, download the dataset separately and place it in the expected local directory structure.

See `data/README.md` for details.

## Citation

If you use this repository, please cite the capstone paper and reference the NinaPro DB3 dataset.

## License

Add a license file before public sharing if you want others to reuse the code.
