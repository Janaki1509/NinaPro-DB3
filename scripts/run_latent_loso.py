#!/usr/bin/env python3
"""
run_latent_loso.py
==================
Shared Conv1D Latent-Space Encoder + Classifier for NinaPro DB3 LOSO benchmark.

DATA LOADING: Identical to run_mlp_loso.py — reads raw .mat files,
windows on the fly, balanced subsample, channel-norm from train set.

New additions over MLP:
  - Conv1D encoder operates on raw (C,T) windows — no flattening
  - 128-dim latent bottleneck
  - Latent vectors saved as .npz for t-SNE visualization
  - JSON output format identical to MLP (compatible with merge script)

Usage (via sbatch ONLY — never on login node):
  python run_latent_loso.py \
      --base /mnt/home/chandraj/ninapro_db3/subjects \
      --test s1 \
      --out_json $HOME/ninapro_db3/runs/latent/s1_latent.json \
      --latent_npz $HOME/ninapro_db3/runs/latent/s1_latents.npz
"""

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Safety: refuse to run on HPC login node
if socket.gethostname().startswith("port"):
    print("ERROR: on login node. Submit via sbatch only.", flush=True)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — identical to run_mlp_loso.py
# ─────────────────────────────────────────────────────────────────────────────

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
            found = _search_for_key(v, target_keys, depth + 1, max_depth)
            if found is not None:
                return found
    if hasattr(obj, "__dict__"):
        d = obj.__dict__
        for k, v in d.items():
            if k in target_keys:
                arr = _maybe_array(v)
                if arr is not None:
                    return arr
        for _, v in d.items():
            found = _search_for_key(v, target_keys, depth + 1, max_depth)
            if found is not None:
                return found
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            found = _search_for_key(item, target_keys, depth + 1, max_depth)
            if found is not None:
                return found
    return None


def load_subject_arrays(subject_dir: Path):
<<<<<<< HEAD
    mats = sorted(subject_dir.glob("*.mat"))
=======
    mats = sorted(subject_dir.rglob("*.mat*"))
>>>>>>> 9bbfa3b24bb49262e519b5672b0b6636e2cc4682
    if not mats:
        raise FileNotFoundError(f"No .mat files in {subject_dir}")
    mat_path = mats[0]
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    emg          = _search_for_key(data, {"emg"})
    restimulus   = _search_for_key(data, {"restimulus"})
    rerepetition = _search_for_key(data, {"rerepetition"})
    if emg is None or restimulus is None or rerepetition is None:
        raise RuntimeError(f"Could not find emg/restimulus/rerepetition in {mat_path}")
    emg          = np.asarray(emg)
    restimulus   = np.asarray(restimulus).reshape(-1)
    rerepetition = np.asarray(rerepetition).reshape(-1)
    if emg.ndim != 2:
        raise RuntimeError(f"Expected 2D emg, got {emg.shape}")
    return emg.astype(np.float32), restimulus.astype(np.int64), rerepetition.astype(np.int64)


def find_homogeneous_segments(labels, reps):
    n = len(labels)
    start = 0
    segments = []
    while start < n:
        lab = int(labels[start])
        rep = int(reps[start])
        end = start + 1
        while end < n and int(labels[end]) == lab and int(reps[end]) == rep:
            end += 1
        if lab != 0:
            segments.append((start, end, lab, rep))
        start = end
    return segments


def make_windows(emg, labels, reps, win_len, step):
    segments = find_homogeneous_segments(labels, reps)
    X, y = [], []
    for start, end, lab, _ in segments:
        if end - start < win_len:
            continue
        for s in range(start, end - win_len + 1, step):
            X.append(emg[s:s + win_len].T)  # (C, T)
            y.append(lab)
    if not X:
        return np.empty((0, emg.shape[1], win_len), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


def balanced_subsample(X, y, per_class, rng):
    idx_all = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        chosen = rng.choice(idx, size=min(len(idx), per_class), replace=False)
        idx_all.append(chosen)
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


def keep_common_classes(X_train, y_train, X_test, y_test):
    common = np.intersect1d(np.unique(y_train), np.unique(y_test))
    return (X_train[np.isin(y_train, common)], y_train[np.isin(y_train, common)],
            X_test[np.isin(y_test, common)],   y_test[np.isin(y_test, common)], common)


def compute_channel_norm(X):
    mean = X.mean(axis=(0, 2), keepdims=True)
    std  = X.std(axis=(0, 2), keepdims=True)
    std  = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ConvLatentClassifier(nn.Module):
    """
    Conv1D shared encoder -> 128-dim latent -> softmax classifier.
    Input:  (batch, C=12, T=300)
    Latent: (batch, latent_dim)
    Output: (batch, n_classes), latent
    """
    def __init__(self, n_channels, win_len, latent_dim=128, n_classes=52, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        flat_size = 128 * (win_len // 8)
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, x):
        z = self.bottleneck(self.encoder(x))
        return self.classifier(z), z


# ─────────────────────────────────────────────────────────────────────────────
# Train / predict
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, loader, epochs, lr, weight_decay, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for epoch in range(epochs):
        total_loss, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            n_seen += len(yb)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch={epoch+1:02d} loss={total_loss/max(n_seen,1):.4f}", flush=True)


@torch.no_grad()
def predict_model(model, X, batch_size, device):
    model.eval()
    all_preds, all_latents = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        logits, z = model(xb)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_latents.append(z.cpu().numpy())
    preds   = np.concatenate(all_preds)   if all_preds   else np.empty((0,), dtype=np.int64)
    latents = np.concatenate(all_latents) if all_latents else np.empty((0, 128), dtype=np.float32)
    return preds, latents


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",            type=str,   required=True)
    ap.add_argument("--test",            type=str,   required=True)
    ap.add_argument("--out_json",        type=str,   required=True)
    ap.add_argument("--latent_npz",      type=str,   default="")
    ap.add_argument("--fs",              type=int,   default=2000)
    ap.add_argument("--win_ms",          type=int,   default=150)
    ap.add_argument("--overlap",         type=float, default=0.5)
    ap.add_argument("--train_per_class", type=int,   default=120)
    ap.add_argument("--test_per_class",  type=int,   default=80)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--epochs",          type=int,   default=50)
    ap.add_argument("--batch_size",      type=int,   default=256)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--weight_decay",    type=float, default=1e-4)
    ap.add_argument("--latent_dim",      type=int,   default=128)
    ap.add_argument("--dropout",         type=float, default=0.3)
    ap.add_argument("--cpus",            type=int,   default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, args.cpus))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = Path(args.base)
    subject_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("s")])
    if not subject_dirs:
        raise RuntimeError(f"No subject dirs under {base}")

    win_len = int(args.fs * args.win_ms / 1000)
    step    = max(1, int(win_len * (1.0 - args.overlap)))

    X_train_all, y_train_all = [], []
    X_test_all,  y_test_all  = [], []

    for subj_dir in subject_dirs:
        emg, labels, reps = load_subject_arrays(subj_dir)
        X_subj, y_subj = make_windows(emg, labels, reps, win_len=win_len, step=step)
        if subj_dir.name == args.test:
            X_test_all.append(X_subj)
            y_test_all.append(y_subj)
        else:
            X_train_all.append(X_subj)
            y_train_all.append(y_subj)

    if not X_train_all or not X_test_all:
        raise RuntimeError("Train/test split failed")

    X_train = np.concatenate(X_train_all)
    y_train = np.concatenate(y_train_all)
    X_test  = np.concatenate(X_test_all)
    y_test  = np.concatenate(y_test_all)

    X_train, y_train, X_test, y_test, common_classes = keep_common_classes(
        X_train, y_train, X_test, y_test)
    if len(common_classes) == 0:
        raise RuntimeError("No common classes between train and test")

    std_ratio = float(X_test.std() / (X_train.std() + 1e-12))

    X_train, y_train = balanced_subsample(X_train, y_train, args.train_per_class, rng)
    X_test,  y_test  = balanced_subsample(X_test,  y_test,  args.test_per_class,  rng)

    cls_sorted  = np.sort(np.unique(np.concatenate([y_train, y_test])))
    cls_to_idx  = {int(c): i for i, c in enumerate(cls_sorted)}
    y_train_idx = np.asarray([cls_to_idx[int(c)] for c in y_train], dtype=np.int64)
    y_test_idx  = np.asarray([cls_to_idx[int(c)] for c in y_test],  dtype=np.int64)

    mean, std = compute_channel_norm(X_train)
    X_train = apply_norm(X_train, mean, std)
    X_test  = apply_norm(X_test,  mean, std)

    N_train, C, T = X_train.shape
    N_test = X_test.shape[0]
    K = len(cls_sorted)
    print(f"{args.test} | K={K} std_ratio={std_ratio:.3f} train={N_train} test={N_test} device={device}", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_idx))
    loader   = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ConvLatentClassifier(n_channels=C, win_len=T, latent_dim=args.latent_dim,
                                  n_classes=K, dropout=args.dropout).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    train_model(model, loader, args.epochs, args.lr, args.weight_decay, device)

    y_pred_idx, latent_vecs = predict_model(model, X_test, args.batch_size, device)

    acc      = float(accuracy_score(y_test_idx, y_pred_idx))
    bacc     = float(balanced_accuracy_score(y_test_idx, y_pred_idx))
    macro_f1 = float(f1_score(y_test_idx, y_pred_idx, average="macro", zero_division=0))
    print(f"{args.test} | acc={acc:.4f} bacc={bacc:.4f} f1={macro_f1:.4f}", flush=True)

    if args.latent_npz:
        Path(args.latent_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.latent_npz,
            latents=latent_vecs.astype(np.float32),
            labels=y_test_idx.astype(np.int32),
            gesture_ids=y_test.astype(np.int32),
            subject=np.array([int(args.test.replace("s", ""))]),
        )
        print(f"  Latents saved: {args.latent_npz}", flush=True)

    out = {
        "method": "latent_cnn",
        "test_subject": args.test,
        "N_train": int(N_train),
        "N_test":  int(N_test),
        "K": int(K),
        "win_ms": int(args.win_ms),
        "overlap": float(args.overlap),
        "strict_purity": True,
        "train_per_class": int(args.train_per_class),
        "test_per_class":  int(args.test_per_class),
        "std_ratio": std_ratio,
        "acc": acc,
        "balanced_acc": bacc,
        "macro_f1": macro_f1,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "latent_dim": int(args.latent_dim),
        "dropout": float(args.dropout),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
