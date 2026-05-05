#!/usr/bin/env python3
"""
run_calibration.py
==================
Confidence calibration analysis for NinaPro DB3 LOSO models.

Runs inference on ONE LOSO fold (default s1), collects softmax probabilities,
fits temperature scaling, and produces:
  1. reliability_diagram_before.png  — uncalibrated
  2. reliability_diagram_after.png   — after temperature scaling
  3. calibration_comparison.png      — side by side
  4. brier_scores.csv                — ECE and Brier before/after

Run on your LAPTOP (no HPC needed) — uses same data loading as MLP script.

Usage:
  python run_calibration.py \
      --base C:/ninapro_db3/subjects \
      --test s1 \
      --model mlp \
      --out_dir C:/ninapro_db3/outputs/calibration
"""

import argparse
import os
import csv
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Data loading (identical to run_mlp_loso.py) ───────────────────────────

def _maybe_array(x):
    try:
        arr = np.asarray(x)
        if arr.size > 0:
            return arr
    except Exception:
        pass
    return None

def _search_for_key(obj, target_keys, depth=0, max_depth=4):
    if depth > max_depth:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in target_keys:
                arr = _maybe_array(v)
                if arr is not None:
                    return arr
        for _, v in obj.items():
            found = _search_for_key(v, target_keys, depth+1, max_depth)
            if found is not None:
                return found
    if hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            if k in target_keys:
                arr = _maybe_array(v)
                if arr is not None:
                    return arr
        for _, v in obj.__dict__.items():
            found = _search_for_key(v, target_keys, depth+1, max_depth)
            if found is not None:
                return found
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            found = _search_for_key(item, target_keys, depth+1, max_depth)
            if found is not None:
                return found
    return None

def load_subject_arrays(subject_dir):
    mats = sorted(Path(subject_dir).glob('*.mat'))
    if not mats:
        raise FileNotFoundError(f'No .mat files in {subject_dir}')
    data = sio.loadmat(str(mats[0]), squeeze_me=True, struct_as_record=False)
    emg          = np.asarray(_search_for_key(data, {'emg'}))
    restimulus   = np.asarray(_search_for_key(data, {'restimulus'})).reshape(-1)
    rerepetition = np.asarray(_search_for_key(data, {'rerepetition'})).reshape(-1)
    return emg.astype(np.float32), restimulus.astype(np.int64), rerepetition.astype(np.int64)

def find_homogeneous_segments(labels, reps):
    n, start, segments = len(labels), 0, []
    while start < n:
        lab, rep, end = int(labels[start]), int(reps[start]), start+1
        while end < n and int(labels[end]) == lab and int(reps[end]) == rep:
            end += 1
        if lab != 0:
            segments.append((start, end, lab, rep))
        start = end
    return segments

def make_windows(emg, labels, reps, win_len, step):
    X, y = [], []
    for start, end, lab, _ in find_homogeneous_segments(labels, reps):
        if end - start < win_len:
            continue
        for s in range(start, end - win_len + 1, step):
            X.append(emg[s:s+win_len].T)
            y.append(lab)
    if not X:
        return np.empty((0, emg.shape[1], win_len), np.float32), np.empty((0,), np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y, np.int64)

def balanced_subsample(X, y, per_class, rng):
    idx_all = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx_all.append(rng.choice(idx, min(len(idx), per_class), replace=False))
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]

def keep_common_classes(Xtr, ytr, Xte, yte):
    common = np.intersect1d(np.unique(ytr), np.unique(yte))
    return (Xtr[np.isin(ytr,common)], ytr[np.isin(ytr,common)],
            Xte[np.isin(yte,common)], yte[np.isin(yte,common)], common)

def compute_channel_norm(X):
    mean = X.mean(axis=(0,2), keepdims=True)
    std  = X.std(axis=(0,2),  keepdims=True)
    return mean.astype(np.float32), np.where(std<1e-6, 1.0, std).astype(np.float32)


# ── Models ────────────────────────────────────────────────────────────────

class MLPSoftmax(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes))
    def forward(self, x):
        return self.net(x)

class ConvLatentClassifier(nn.Module):
    def __init__(self, n_channels, win_len, latent_dim=128, n_classes=52, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.bottleneck = nn.Sequential(
            nn.Flatten(), nn.Linear(128*(win_len//8), latent_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(latent_dim, n_classes)
    def forward(self, x):
        z = self.bottleneck(self.encoder(x))
        return self.classifier(z), z


# ── Temperature scaling ───────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    def forward(self, logits):
        return logits / self.temperature

def fit_temperature(logits, labels, lr=0.01, max_iter=200):
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(labels).long()
    def eval_fn():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_t), labels_t)
        loss.backward()
        return loss
    optimizer.step(eval_fn)
    T = float(scaler.temperature.item())
    print(f'  Optimal temperature: {T:.4f}')
    return T


# ── Calibration metrics ───────────────────────────────────────────────────

def softmax_np(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def brier_score(probs, labels):
    n, K = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot)**2, axis=1)))

def expected_calibration_error(probs, labels, n_bins=10):
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(labels))

def plot_reliability(probs, labels, title, out_path, n_bins=10):
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    bin_acc, bin_conf, bin_count = [], [], []
    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if mask.sum() == 0:
            bin_acc.append(0); bin_conf.append((bins[i]+bins[i+1])/2); bin_count.append(0)
        else:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confs[mask].mean())
            bin_count.append(mask.sum())
    ece  = expected_calibration_error(probs, labels, n_bins)
    bs   = brier_score(probs, labels)
    bacc = balanced_accuracy_score(labels, preds)
    bin_centers = [(bins[i]+bins[i+1])/2 for i in range(n_bins)]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0,1],[0,1],'--', color='#aaa', linewidth=1, label='Perfect calibration')
    ax.bar(bin_centers, bin_acc, width=0.09, alpha=0.7, color='#5b7fce',
           edgecolor='white', label='Model confidence', zorder=3)
    ax.bar(bin_centers, [b/max(bin_count) for b in bin_count],
           width=0.09, alpha=0.2, color='#e06050', label='Sample density')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{title}\nECE={ece:.4f}  Brier={bs:.4f}  bAcc={bacc:.4f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    return ece, bs, bacc


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',     required=True)
    ap.add_argument('--test',     default='s1')
    ap.add_argument('--model',    default='mlp', choices=['mlp','latent'])
    ap.add_argument('--out_dir',  default='./outputs/calibration')
    ap.add_argument('--fs',       type=int,   default=2000)
    ap.add_argument('--win_ms',   type=int,   default=150)
    ap.add_argument('--overlap',  type=float, default=0.5)
    ap.add_argument('--train_per_class', type=int, default=120)
    ap.add_argument('--test_per_class',  type=int, default=80)
    ap.add_argument('--epochs',   type=int,   default=30)
    ap.add_argument('--seed',     type=int,   default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cpu')

    base = Path(args.base)
    subject_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith('s')])
    win_len = int(args.fs * args.win_ms / 1000)
    step    = max(1, int(win_len * (1 - args.overlap)))

    X_train_all, y_train_all, X_test_all, y_test_all = [], [], [], []
    for sd in subject_dirs:
        emg, labels, reps = load_subject_arrays(sd)
        X, y = make_windows(emg, labels, reps, win_len, step)
        if sd.name == args.test:
            X_test_all.append(X); y_test_all.append(y)
        else:
            X_train_all.append(X); y_train_all.append(y)

    X_train = np.concatenate(X_train_all); y_train = np.concatenate(y_train_all)
    X_test  = np.concatenate(X_test_all);  y_test  = np.concatenate(y_test_all)
    X_train, y_train, X_test, y_test, _ = keep_common_classes(X_train, y_train, X_test, y_test)

    X_train, y_train = balanced_subsample(X_train, y_train, args.train_per_class, rng)

    # Split train into train (80%) + val (20%) for calibration fitting
    n = len(X_train)
    val_idx   = rng.choice(n, int(n*0.2), replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)
    X_val, y_val   = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test,  y_test  = balanced_subsample(X_test, y_test, args.test_per_class, rng)

    cls_sorted = np.sort(np.unique(np.concatenate([y_train, y_val, y_test])))
    c2i = {int(c): i for i, c in enumerate(cls_sorted)}
    y_train_i = np.array([c2i[int(c)] for c in y_train], np.int64)
    y_val_i   = np.array([c2i[int(c)] for c in y_val],   np.int64)
    y_test_i  = np.array([c2i[int(c)] for c in y_test],  np.int64)

    mean, std = compute_channel_norm(X_train)
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    N, C, T = X_train.shape
    K = len(cls_sorted)
    print(f'Train={N} Val={len(X_val)} Test={len(X_test)} K={K}')

    # ── Train model ──
    if args.model == 'mlp':
        X_tr_in = torch.from_numpy(X_train.reshape(N, C*T))
        X_va_in = torch.from_numpy(X_val.reshape(len(X_val), C*T))
        X_te_in = torch.from_numpy(X_test.reshape(len(X_test), C*T))
        model = MLPSoftmax(C*T, 128, K, dropout=0.2).to(device)
        model_name = 'MLP SoftMax'
    else:
        X_tr_in = torch.from_numpy(X_train)
        X_va_in = torch.from_numpy(X_val)
        X_te_in = torch.from_numpy(X_test)
        model = ConvLatentClassifier(C, T, 128, K, 0.3).to(device)
        model_name = 'Latent CNN'

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr_in, torch.from_numpy(y_train_i)),
                        batch_size=256, shuffle=True)

    print(f'Training {model_name} for {args.epochs} epochs...')
    model.train()
    for ep in range(args.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            logits = out[0] if isinstance(out, tuple) else out
            criterion(logits, yb).backward()
            optimizer.step()
        if (ep+1) % 10 == 0:
            print(f'  epoch {ep+1}', flush=True)

    # ── Get logits on val (for temperature fitting) and test ──
    model.eval()
    with torch.no_grad():
        out_val  = model(X_va_in)
        out_test = model(X_te_in)
    logits_val  = (out_val[0]  if isinstance(out_val,  tuple) else out_val).numpy()
    logits_test = (out_test[0] if isinstance(out_test, tuple) else out_test).numpy()

    # ── Before calibration ──
    probs_before = softmax_np(logits_test)
    ece_b, bs_b, bacc_b = plot_reliability(
        probs_before, y_test_i,
        f'{model_name} — before calibration (test subject {args.test})',
        os.path.join(args.out_dir, f'reliability_before_{args.model}_{args.test}.png'))
    print(f'Before — ECE={ece_b:.4f}  Brier={bs_b:.4f}  bAcc={bacc_b:.4f}')

    # ── Fit temperature on validation set ──
    print('Fitting temperature scaling on validation set...')
    T_opt = fit_temperature(logits_val, y_val_i)

    # ── After calibration ──
    probs_after = softmax_np(logits_test / T_opt)
    ece_a, bs_a, bacc_a = plot_reliability(
        probs_after, y_test_i,
        f'{model_name} — after temperature scaling (T={T_opt:.2f})',
        os.path.join(args.out_dir, f'reliability_after_{args.model}_{args.test}.png'))
    print(f'After  — ECE={ece_a:.4f}  Brier={bs_a:.4f}  bAcc={bacc_a:.4f}')

    # ── Side-by-side comparison figure ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, probs, title, ece, bs in [
        (axes[0], probs_before, 'Before calibration', ece_b, bs_b),
        (axes[1], probs_after,  f'After temp scaling (T={T_opt:.2f})', ece_a, bs_a)
    ]:
        confs  = probs.max(axis=1)
        preds  = probs.argmax(axis=1)
        correct = (preds == y_test_i).astype(float)
        bins = np.linspace(0, 1, 11)
        bin_centers = [(bins[i]+bins[i+1])/2 for i in range(10)]
        bin_acc = []
        for i in range(10):
            mask = (confs >= bins[i]) & (confs < bins[i+1])
            bin_acc.append(correct[mask].mean() if mask.sum() > 0 else 0)
        ax.plot([0,1],[0,1],'--',color='#aaa',linewidth=1.2)
        ax.bar(bin_centers, bin_acc, width=0.09, alpha=0.75,
               color='#5b7fce', edgecolor='white', zorder=3)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
        ax.set_title(f'{title}\nECE={ece:.4f}  Brier={bs:.4f}', fontweight='bold', fontsize=10)
        ax.grid(alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.suptitle(f'{model_name} confidence calibration — subject {args.test}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f'calibration_comparison_{args.model}_{args.test}.png'),
                dpi=180, bbox_inches='tight')
    plt.close()
    print(f'Saved comparison plot.')

    # ── Save CSV ──
    csv_path = os.path.join(args.out_dir, 'calibration_results.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['model','subject','temperature','ece_before','brier_before',
                        'bacc_before','ece_after','brier_after','bacc_after'])
        w.writerow([args.model, args.test, f'{T_opt:.4f}',
                    f'{ece_b:.4f}', f'{bs_b:.4f}', f'{bacc_b:.4f}',
                    f'{ece_a:.4f}', f'{bs_a:.4f}', f'{bacc_a:.4f}'])
    print(f'Results saved to {csv_path}')
    print(f'\nKey finding: calibration {"improved" if ece_a < ece_b else "did not improve"} ECE '
          f'from {ece_b:.4f} to {ece_a:.4f}')

if __name__ == '__main__':
    main()
