readme_content = """# NinaPro DB3: Reproducible Cross-Subject EMG Gesture Classification

Reproducible benchmark for cross-subject surface EMG gesture recognition on NinaPro DB3, with leave-one-subject-out evaluation, domain shift analysis, confidence calibration, and comparison of multiple baseline and neural models.

## Project Overview

This project studies how well EMG gesture-recognition models generalize to unseen amputee subjects. Instead of relying on within-subject performance, the benchmark uses leave-one-subject-out (LOSO) evaluation to simulate a realistic clinical setting.

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

Published EMG results often look strong under within-subject testing, but performance drops sharply when the model must predict for a new subject. That gap matters for prosthetic control, where a system must generalize to a new patient rather than memorize one person’s signal pattern.

## Dataset

- **Dataset:** NinaPro DB3
- **Subjects:** 11 transradial amputees
- **Signals:** 12 sEMG channels
- **Primary task:** Exercise 1, 17 gesture classes
- **Preprocessing:** 150 ms windows, 50% overlap, REST removed, channel-wise normalization

## Data Instructions

This repository does not include the raw NinaPro DB3 dataset.

### Where to get the data
Download NinaPro DB3 from the official NinaPro project page and follow its access instructions.

### Local folder setup
Place the dataset in the local directory structure expected by the scripts in this repository.

Example:

```bash
ninapro_db3/
├── data/
├── folds/
├── results/
├── scripts/
├── sbatch/
└── subjects/
```

### Notes
- Do not commit the raw dataset to GitHub.
- Large binary files such as `.mat` and `.npz` files are intentionally excluded from version control.
- The repository contains code, job scripts, results, and documentation needed to reproduce the experiments.

## Repo Structure

- `scripts/` — Python scripts for training, evaluation, plotting, and result processing
- `sbatch/` — Slurm job scripts used on Clipper HPC
- `results/` — CSV summaries and per-subject outputs
- `docs/` — capstone paper, poster, and summary files
- `figures/` — final figures for the report and LinkedIn
- `data/README.md` — dataset instructions for local setup

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

## How to Run on Clipper

This project was developed and tested on the Clipper HPC cluster at GVSU.

### 1. Connect to Clipper
Use SSH to log in to Clipper:

```bash
ssh yourNetID@clipper.gvsu.edu
```

### 2. Go to the project directory
```bash
cd ~/ninapro_db3
```

### 3. Activate your Python environment
```bash
source ~/venvs/ninapro/bin/activate
```

If you use a different environment name, update the command accordingly.

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run a preprocessing or training script
Examples:

```bash
python scripts/run_mlp_loso.py
python scripts/run_latent_loso.py
python scripts/run_cycle_autoencoder_loso.py
```

### 6. Submit Slurm jobs
For the larger experiments, submit the batch scripts in the `sbatch/` folder:

```bash
sbatch sbatch/run_mlp_loso_array.sbatch
sbatch sbatch/run_latent_loso_array.sbatch
sbatch sbatch/run_cycle_loso_array.sbatch
```

### 7. Merge and summarize results
After jobs finish, merge per-subject outputs and create summary tables:

```bash
python scripts/merge_per_subject_json_to_csv.py
python scripts/summarize_results.py
```

### 8. Generate final plots
```bash
python make_final_plots.py
```

## References

1. Atzori M, et al. Electromyography data for non-invasive naturally controlled robotic hand prostheses. *Scientific Data*. 2014;1:140053.
2. NinaPro Project. NinaPro DB3. https://ninapro.hevs.ch/instructions/DB3.html.
3. Niu Q, et al. Motion intention recognition of the affected hand based on sEMG and improved DenseNet. *Heliyon*. 2024;10:e26763.
4. Ovadia D, Segal A, Rabin N. Classification of hand and wrist movements via sEMG using random convolutional kernels. *Scientific Reports*. 2024;14:4134.
5. Sandoval-Espino JA, et al. Selection of the best set of features for sEMG-based hand gesture recognition applying to a CNN. *Sensors*. 2022;22:4972.
6. Dempster A, et al. MiniROCKET: A very fast time series classifier. *KDD*. 2021:248-257.
7. Paszke A, et al. PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*. 2019;32.
8. Guo C, et al. On calibration of modern neural networks. *ICML*. 2017.
9. Gretton A, et al. A kernel two sample test. *JMLR*. 2012;13:723-773.

## Citation

If you use this repository, please cite the capstone paper:

```bibtex
@misc{chandrapalakal2026ninaprodb3,
  author       = {Chandrapalakal, Janaki},
  title        = {Reproducible Cross-Subject EMG Gesture Classification on NinaPro DB3: Benchmark Evaluation, Domain Adaptation, Confidence Calibration, and Agentic Search},
  year         = {2026},
  institution  = {Grand Valley State University},
  note         = {CIS 691 Capstone Project}
}
```

"""

with open('README.md', 'w') as f:
    f.write(readme_content)
"""

with open('data/README.md', 'w') as f:
    f.write("# Data Instructions\n\nThis repository does not include the raw NinaPro DB3 dataset.\n\n## Dataset\n- **Dataset name:** NinaPro DB3\n- **Description:** Surface EMG recordings from transradial amputee subjects\n- **Primary use in this project:** Cross-subject gesture classification with leave-one-subject-out evaluation\n\n## Where to get the data\nDownload NinaPro DB3 from the official NinaPro project page and follow its access instructions.\n\n## Local folder setup\nPlace the dataset in the local directory structure expected by the scripts in this repository.\n\n## Notes\n- Do not commit the raw dataset to GitHub.\n- Large binary files such as `.mat` and `.npz` files are intentionally excluded from version control.\n- The repository contains code, job scripts, results, and documentation needed to reproduce the experiments.")
"""
