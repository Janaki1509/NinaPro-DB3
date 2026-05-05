#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Helpers for reading .mat data
# ----------------------------
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
    mats = sorted(subject_dir.glob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {subject_dir}")

    mat_path = mats[0]
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    emg = _search_for_key(data, {"emg"})
    restimulus = _search_for_key(data, {"restimulus"})
    rerepetition = _search_for_key(data, {"rerepetition"})

    if emg is None or restimulus is None or rerepetition is None:
        raise RuntimeError(
            f"Could not find emg/restimulus/rerepetition in {mat_path}"
        )

    emg = np.asarray(emg)
    restimulus = np.asarray(restimulus).reshape(-1)
    rerepetition = np.asarray(rerepetition).reshape(-1)

    if emg.ndim != 2:
        raise RuntimeError(f"Expected emg to be 2D, got shape {emg.shape}")

    if emg.shape[0] != restimulus.shape[0] or emg.shape[0] != rerepetition.shape[0]:
        raise RuntimeError(
            f"Length mismatch in {mat_path}: emg={emg.shape}, "
            f"restimulus={restimulus.shape}, rerepetition={rerepetition.shape}"
        )

    return emg.astype(np.float32), restimulus.astype(np.int64), rerepetition.astype(np.int64), str(mat_path)


# ----------------------------
# Windowing
# ----------------------------
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
        # skip REST
        if lab != 0:
            segments.append((start, end, lab, rep))
        start = end
    return segments


def make_windows(emg, labels, reps, win_len, step):
    segments = find_homogeneous_segments(labels, reps)
    X, y = [], []

    for start, end, lab, _ in segments:
        seg_len = end - start
        if seg_len < win_len:
            continue
        for s in range(start, end - win_len + 1, step):
            w = emg[s:s + win_len]  # (T, C)
            X.append(w.T)           # store as (C, T)
            y.append(lab)

    if not X:
        return np.empty((0, emg.shape[1], win_len), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


# ----------------------------
# Balanced subsampling
# ----------------------------
def balanced_subsample(X, y, per_class, rng):
    classes = np.unique(y)
    idx_all = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        take = min(len(idx), per_class)
        chosen = rng.choice(idx, size=take, replace=False)
        idx_all.append(chosen)
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


def keep_common_classes(X_train, y_train, X_test, y_test):
    common = np.intersect1d(np.unique(y_train), np.unique(y_test))
    tr_mask = np.isin(y_train, common)
    te_mask = np.isin(y_test, common)
    return X_train[tr_mask], y_train[tr_mask], X_test[te_mask], y_test[te_mask], common


# ----------------------------
# Normalization
# ----------------------------
def compute_channel_norm(X):
    # X shape: (N, C, T)
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


# ----------------------------
# Model
# ----------------------------
class MLPSoftmax(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, epochs, lr, weight_decay, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            total_loss += loss.item() * bs
            n_seen += bs

        avg_loss = total_loss / max(n_seen, 1)
        print(f"epoch={epoch+1:02d} loss={avg_loss:.4f}", flush=True)


@torch.no_grad()
def predict_model(model, X, batch_size, device):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)


# ----------------------------
# Main
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True, help="Path containing subject folders s1, s2, ...")
    ap.add_argument("--test", type=str, required=True, help="Test subject, e.g. s1")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--tmp", type=str, default="")
    ap.add_argument("--fs", type=int, default=2000)
    ap.add_argument("--win_ms", type=int, default=150)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--train_per_class", type=int, default=120)
    ap.add_argument("--test_per_class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    torch.set_num_threads(max(1, int(args.cpus)))
    device = torch.device("cpu")

    base = Path(args.base)
    test_subject = args.test
    subject_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("s")])

    if not subject_dirs:
        raise RuntimeError(f"No subject directories found under {base}")

    win_len = int(args.fs * args.win_ms / 1000)
    step = max(1, int(win_len * (1.0 - args.overlap)))

    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []

    for subj_dir in subject_dirs:
        emg, labels, reps, mat_path = load_subject_arrays(subj_dir)
        X_subj, y_subj = make_windows(emg, labels, reps, win_len=win_len, step=step)

        if subj_dir.name == test_subject:
            X_test_all.append(X_subj)
            y_test_all.append(y_subj)
        else:
            X_train_all.append(X_subj)
            y_train_all.append(y_subj)

    if not X_train_all or not X_test_all:
        raise RuntimeError("Train/test split failed; check subject folders and --test argument")

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)

    X_train, y_train, X_test, y_test, common_classes = keep_common_classes(X_train, y_train, X_test, y_test)

    if len(common_classes) == 0:
        raise RuntimeError("No common gesture classes between train and test")

    std_ratio = float(X_test.std() / (X_train.std() + 1e-12))

    X_train, y_train = balanced_subsample(X_train, y_train, args.train_per_class, rng)
    X_test, y_test = balanced_subsample(X_test, y_test, args.test_per_class, rng)

    # remap labels to 0..K-1
    cls_sorted = np.sort(np.unique(np.concatenate([y_train, y_test])))
    cls_to_idx = {int(c): i for i, c in enumerate(cls_sorted)}
    y_train_idx = np.asarray([cls_to_idx[int(c)] for c in y_train], dtype=np.int64)
    y_test_idx = np.asarray([cls_to_idx[int(c)] for c in y_test], dtype=np.int64)

    mean, std = compute_channel_norm(X_train)
    X_train = apply_norm(X_train, mean, std)
    X_test = apply_norm(X_test, mean, std)

    N_train, C, T = X_train.shape
    N_test = X_test.shape[0]
    K = len(cls_sorted)

    X_train_flat = X_train.reshape(N_train, C * T).astype(np.float32)
    X_test_flat = X_test.reshape(N_test, C * T).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train_flat), torch.from_numpy(y_train_idx))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = MLPSoftmax(
        input_dim=C * T,
        hidden_dim=args.hidden_dim,
        n_classes=K,
        dropout=args.dropout,
    ).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
    )

    y_pred_idx = predict_model(model, X_test_flat, batch_size=args.batch_size, device=device)

    acc = float(accuracy_score(y_test_idx, y_pred_idx))
    bacc = float(balanced_accuracy_score(y_test_idx, y_pred_idx))
    macro_f1 = float(f1_score(y_test_idx, y_pred_idx, average="macro", zero_division=0))

    out = {
        "method": "mlp_softmax",
        "test_subject": test_subject,
        "N_train": int(N_train),
        "N_test": int(N_test),
        "K": int(K),
        "win_ms": int(args.win_ms),
        "overlap": float(args.overlap),
        "strict_purity": True,
        "train_per_class": int(args.train_per_class),
        "test_per_class": int(args.test_per_class),
        "std_ratio": std_ratio,
        "acc": acc,
        "balanced_acc": bacc,
        "macro_f1": macro_f1,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"{test_subject} | K={K} | std_ratio={std_ratio:.3f} | "
        f"acc={acc:.3f} | bacc={bacc:.3f} | f1={macro_f1:.3f}"
    )
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
PY