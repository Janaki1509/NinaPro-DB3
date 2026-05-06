<<<<<<< HEAD
#!/usr/bin/env python3
"""
run_conditional_latent_loso.py
================================
Conditional Latent CNN for NinaPro DB3 LOSO benchmark.

What is new vs run_latent_loso.py:
  - Subject one-hot vector appended to 128-dim latent before classifier
  - Latent vector becomes 128 + N_subjects = 139-dim
  - Everything else identical: same data loading, same LOSO protocol,
    same JSON output, same sbatch workflow

Professor requested: conditional modeling (Action Item 1)
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

# Safety: never run on login node
if socket.gethostname().startswith("port"):
    print("ERROR: on login node. Submit via sbatch only.", flush=True)
    sys.exit(1)


# ── Data loading (identical to run_latent_loso.py) ────────────────────────

def _maybe_array(x):
    try:
        arr = np.asarray(x)
        if arr.size > 0:
            return arr
    except Exception:
        pass
    return None

def _search(obj, keys, depth=0, max_depth=4):
    if depth > max_depth:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                a = _maybe_array(v)
                if a is not None:
                    return a
        for _, v in obj.items():
            f = _search(v, keys, depth+1, max_depth)
            if f is not None:
                return f
    if hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            if k in keys:
                a = _maybe_array(v)
                if a is not None:
                    return a
        for _, v in obj.__dict__.items():
            f = _search(v, keys, depth+1, max_depth)
            if f is not None:
                return f
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            f = _search(item, keys, depth+1, max_depth)
            if f is not None:
                return f
    return None

def load_subject(sd):
    mats = sorted(Path(sd).rglob('*.mat.mat')) or sorted(Path(sd).rglob('*.mat'))
    if not mats:
        raise FileNotFoundError(f'No .mat files in {sd}')
    data = sio.loadmat(str(mats[0]), squeeze_me=True, struct_as_record=False)
    emg  = np.asarray(_search(data, {'emg'}))
    stim = np.asarray(_search(data, {'restimulus'})).reshape(-1)
    rep  = np.asarray(_search(data, {'rerepetition'})).reshape(-1)
    return emg.astype(np.float32), stim.astype(np.int64), rep.astype(np.int64)

def make_windows(emg, labels, reps, win_len=300, step=150):
    n, start, segs = len(labels), 0, []
    while start < n:
        lab, rep, end = int(labels[start]), int(reps[start]), start + 1
        while end < n and int(labels[end]) == lab and int(reps[end]) == rep:
            end += 1
        if lab != 0:
            segs.append((start, end, lab))
        start = end
    X, y = [], []
    for s, e, lab in segs:
        if e - s < win_len:
            continue
        for i in range(s, e - win_len + 1, step):
            X.append(emg[i:i+win_len].T)
            y.append(lab)
    if not X:
        return np.empty((0, emg.shape[1], win_len), np.float32), np.empty((0,), np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y, np.int64)

def subsample(X, y, n, rng):
    idx = []
    for c in np.unique(y):
        i = np.flatnonzero(y == c)
        idx.append(rng.choice(i, min(len(i), n), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]

def common_classes(Xtr, ytr, Xte, yte):
    c = np.intersect1d(np.unique(ytr), np.unique(yte))
    return (Xtr[np.isin(ytr, c)], ytr[np.isin(ytr, c)],
            Xte[np.isin(yte, c)], yte[np.isin(yte, c)], c)

def channel_norm(X):
    m = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    s = X.std(axis=(0, 2),  keepdims=True)
    return m, np.where(s < 1e-6, 1.0, s).astype(np.float32)

def remap(y_tr, y_te):
    cls = np.sort(np.unique(np.concatenate([y_tr, y_te])))
    c2i = {int(c): i for i, c in enumerate(cls)}
    return (np.array([c2i[int(c)] for c in y_tr], np.int64),
            np.array([c2i[int(c)] for c in y_te], np.int64),
            len(cls))


# ── Model ─────────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """Same Conv1D encoder as Latent CNN — output is 128-dim latent."""
    def __init__(self, n_channels, win_len, latent_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        flat = 128 * (win_len // 8)
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.bottleneck(self.encoder(x))


class ConditionalLatentClassifier(nn.Module):
    """
    Conditional Latent CNN.

    What is new vs plain Latent CNN:
      After encoding x to 128-dim latent z, we concatenate a one-hot
      subject vector (length = n_subjects). The combined vector
      [z | subject_onehot] goes through an extra dense layer, then
      the SoftMax classifier.

      This lets the model adjust its decision boundary per-subject,
      factorising subject-specific variation from gesture variation.
    """
    def __init__(self, n_channels, win_len,
                 latent_dim=128, n_subjects=11, n_classes=52,
                 hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder    = ConvEncoder(n_channels, win_len, latent_dim, dropout)
        # combined dim = latent + subject one-hot
        combined_dim    = latent_dim + n_subjects
        self.head       = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.n_subjects = n_subjects

    def forward(self, x, subj_onehot):
        z       = self.encoder(x)                  # (B, latent_dim)
        z_cond  = torch.cat([z, subj_onehot], dim=1)  # (B, latent+n_subj)
        return self.head(z_cond), z                # logits, latent


# ── Train / eval ──────────────────────────────────────────────────────────

def train_model(model, loader, epochs, lr, wd, device):
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(epochs):
        for xb, sb, yb in loader:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb, sb)
            crit(logits, yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  epoch={ep+1:02d}', flush=True)

@torch.no_grad()
def predict_model(model, X, S, batch_size, device):
    model.eval()
    preds, latents = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device)
        sb = torch.from_numpy(S[i:i+batch_size]).to(device)
        logits, z = model(xb, sb)
        preds.append(logits.argmax(1).cpu().numpy())
        latents.append(z.cpu().numpy())
    return (np.concatenate(preds)   if preds   else np.empty((0,), np.int64),
            np.concatenate(latents) if latents else np.empty((0, 128), np.float32))


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',            required=True)
    ap.add_argument('--test',            required=True,  help='e.g. s1')
    ap.add_argument('--out_json',        required=True)
    ap.add_argument('--latent_npz',      default='')
    ap.add_argument('--n_subjects',      type=int,   default=11)
    ap.add_argument('--fs',              type=int,   default=2000)
    ap.add_argument('--win_ms',          type=int,   default=150)
    ap.add_argument('--overlap',         type=float, default=0.5)
    ap.add_argument('--train_per_class', type=int,   default=120)
    ap.add_argument('--test_per_class',  type=int,   default=80)
    ap.add_argument('--seed',            type=int,   default=42)
    ap.add_argument('--epochs',          type=int,   default=50)
    ap.add_argument('--batch_size',      type=int,   default=256)
    ap.add_argument('--lr',              type=float, default=1e-3)
    ap.add_argument('--weight_decay',    type=float, default=1e-4)
    ap.add_argument('--latent_dim',      type=int,   default=128)
    ap.add_argument('--hidden_dim',      type=int,   default=128)
    ap.add_argument('--dropout',         type=float, default=0.3)
    ap.add_argument('--cpus',            type=int,   default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, args.cpus))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base = Path(args.base)
    sds  = sorted([p for p in base.iterdir()
                   if p.is_dir() and p.name.startswith('s')],
                  key=lambda x: int(x.name.replace('s', '')))
    if not sds:
        raise RuntimeError(f'No subject dirs under {base}')

    # Build subject index map  s1→0, s2→1, ...
    subj_names = [p.name for p in sds]
    subj2idx   = {n: i for i, n in enumerate(subj_names)}
    N_SUBJ     = len(sds)

    win_len = int(args.fs * args.win_ms / 1000)   # 300
    step    = max(1, int(win_len * (1 - args.overlap)))  # 150

    # Load all subjects
    Xtr_all, ytr_all, str_all = [], [], []
    Xte_all, yte_all          = [], []

    for sd in sds:
        emg, lbl, rep = load_subject(sd)
        X, y = make_windows(emg, lbl, rep, win_len, step)
        sidx = subj2idx[sd.name]
        if sd.name == args.test:
            Xte_all.append(X)
            yte_all.append(y)
            # test subject one-hot
            S_te = np.zeros((len(X), N_SUBJ), dtype=np.float32)
            S_te[:, sidx] = 1.0
        else:
            Xtr_all.append(X)
            ytr_all.append(y)
            S_tr = np.zeros((len(X), N_SUBJ), dtype=np.float32)
            S_tr[:, sidx] = 1.0
            str_all.append(S_tr)

    if not Xtr_all or not Xte_all:
        raise RuntimeError('Train/test split failed')

    X_train = np.concatenate(Xtr_all)
    y_train = np.concatenate(ytr_all)
    S_train = np.concatenate(str_all)   # subject one-hots for training
    X_test  = np.concatenate(Xte_all)
    y_test  = np.concatenate(yte_all)
    # test one-hot: all same (the held-out subject)
    test_idx = subj2idx[args.test]
    S_test   = np.zeros((len(X_test), N_SUBJ), dtype=np.float32)
    S_test[:, test_idx] = 1.0

    X_train, y_train, X_test, y_test, _ = common_classes(
        X_train, y_train, X_test, y_test)
    # apply same mask to S_train / S_test
    mask_tr = np.isin(np.concatenate(ytr_all),
                      np.intersect1d(np.unique(y_train), np.unique(y_test)))
    # re-filter properly
    common = np.intersect1d(
        np.unique(np.concatenate(ytr_all)),
        np.unique(np.concatenate(yte_all)))
    Xtr2, ytr2, Str2 = [], [], []
    for X, y, S in zip(Xtr_all, ytr_all, str_all):
        m = np.isin(y, common)
        Xtr2.append(X[m]); ytr2.append(y[m]); Str2.append(S[m])
    Xte2, yte2 = [], []
    for X, y in zip(Xte_all, yte_all):
        m = np.isin(y, common)
        Xte2.append(X[m]); yte2.append(y[m])

    X_train = np.concatenate(Xtr2); y_train = np.concatenate(ytr2)
    S_train = np.concatenate(Str2)
    X_test  = np.concatenate(Xte2); y_test  = np.concatenate(yte2)
    S_test  = np.zeros((len(X_test), N_SUBJ), dtype=np.float32)
    S_test[:, test_idx] = 1.0

    std_ratio = float(X_test.std() / (X_train.std() + 1e-12))

    X_train, y_train = subsample(X_train, y_train, args.train_per_class, rng)
    # subsample S_train to match
    rng2 = np.random.default_rng(args.seed)
    idx_s = []
    for c in np.unique(y_train):
        ii = np.flatnonzero(y_train == c)
        idx_s.append(rng2.choice(ii, min(len(ii), args.train_per_class), replace=False))
    idx_s = np.concatenate(idx_s); rng2.shuffle(idx_s)
    X_train = X_train[idx_s]; y_train = y_train[idx_s]
    S_train = S_train[idx_s]

    X_test, y_test = subsample(X_test, y_test, args.test_per_class, rng)
    idx_te = []
    rng3 = np.random.default_rng(args.seed)
    for c in np.unique(y_test):
        ii = np.flatnonzero(y_test == c)
        idx_te.append(rng3.choice(ii, min(len(ii), args.test_per_class), replace=False))
    idx_te = np.concatenate(idx_te); rng3.shuffle(idx_te)
    X_test  = X_test[idx_te];  y_test  = y_test[idx_te]
    S_test  = S_test[idx_te]

    cls_sorted  = np.sort(np.unique(np.concatenate([y_train, y_test])))
    c2i         = {int(c): i for i, c in enumerate(cls_sorted)}
    y_train_idx = np.array([c2i[int(c)] for c in y_train], np.int64)
    y_test_idx  = np.array([c2i[int(c)] for c in y_test],  np.int64)
    K = len(cls_sorted)

    m, s    = channel_norm(X_train)
    X_train = (X_train - m) / s
    X_test  = (X_test  - m) / s

    N, C, T = X_train.shape
    print(f'{args.test} | K={K} std_ratio={std_ratio:.3f} '
          f'train={N} test={len(X_test)} device={device}', flush=True)

    ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(S_train),
        torch.from_numpy(y_train_idx))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ConditionalLatentClassifier(
        n_channels=C, win_len=T,
        latent_dim=args.latent_dim,
        n_subjects=N_SUBJ,
        n_classes=K,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Params: {n_params:,}', flush=True)

    train_model(model, loader, args.epochs, args.lr, args.weight_decay, device)

    preds, latent_vecs = predict_model(
        model, X_test, S_test, args.batch_size, device)

    acc      = float(accuracy_score(y_test_idx, preds))
    bacc     = float(balanced_accuracy_score(y_test_idx, preds))
    macro_f1 = float(f1_score(y_test_idx, preds, average='macro', zero_division=0))

    print(f'{args.test} | acc={acc:.4f} bacc={bacc:.4f} f1={macro_f1:.4f}', flush=True)

    if args.latent_npz:
        Path(args.latent_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.latent_npz,
            latents=latent_vecs.astype(np.float32),
            labels=y_test_idx.astype(np.int32),
            gesture_ids=y_test.astype(np.int32),
            subject=np.array([int(args.test.replace('s', ''))]),
        )
        print(f'  Latents saved: {args.latent_npz}', flush=True)

    out = {
        'method':        'conditional_latent_cnn',
        'test_subject':  args.test,
        'N_train':       int(N),
        'N_test':        int(len(X_test)),
        'K':             int(K),
        'n_subjects':    int(N_SUBJ),
        'win_ms':        int(args.win_ms),
        'overlap':       float(args.overlap),
        'strict_purity': True,
        'train_per_class': int(args.train_per_class),
        'test_per_class':  int(args.test_per_class),
        'std_ratio':     std_ratio,
        'acc':           acc,
        'balanced_acc':  bacc,
        'macro_f1':      macro_f1,
        'epochs':        int(args.epochs),
        'latent_dim':    int(args.latent_dim),
        'hidden_dim':    int(args.hidden_dim),
        'dropout':       float(args.dropout),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved: {args.out_json}', flush=True)

if __name__ == '__main__':
    main()
=======
#!/usr/bin/env python3
"""
run_conditional_latent_loso.py
================================
Conditional Latent CNN for NinaPro DB3 LOSO benchmark.

What is new vs run_latent_loso.py:
  - Subject one-hot vector appended to 128-dim latent before classifier
  - Latent vector becomes 128 + N_subjects = 139-dim
  - Everything else identical: same data loading, same LOSO protocol,
    same JSON output, same sbatch workflow

Professor requested: conditional modeling (Action Item 1)
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

# Safety: never run on login node
if socket.gethostname().startswith("port"):
    print("ERROR: on login node. Submit via sbatch only.", flush=True)
    sys.exit(1)


# ── Data loading (identical to run_latent_loso.py) ────────────────────────

def _maybe_array(x):
    try:
        arr = np.asarray(x)
        if arr.size > 0:
            return arr
    except Exception:
        pass
    return None

def _search(obj, keys, depth=0, max_depth=4):
    if depth > max_depth:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                a = _maybe_array(v)
                if a is not None:
                    return a
        for _, v in obj.items():
            f = _search(v, keys, depth+1, max_depth)
            if f is not None:
                return f
    if hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            if k in keys:
                a = _maybe_array(v)
                if a is not None:
                    return a
        for _, v in obj.__dict__.items():
            f = _search(v, keys, depth+1, max_depth)
            if f is not None:
                return f
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            f = _search(item, keys, depth+1, max_depth)
            if f is not None:
                return f
    return None

def load_subject(sd):
    mats = sorted(Path(sd).rglob('*.mat.mat')) or sorted(Path(sd).rglob('*.mat'))
    if not mats:
        raise FileNotFoundError(f'No .mat files in {sd}')
    data = sio.loadmat(str(mats[0]), squeeze_me=True, struct_as_record=False)
    emg  = np.asarray(_search(data, {'emg'}))
    stim = np.asarray(_search(data, {'restimulus'})).reshape(-1)
    rep  = np.asarray(_search(data, {'rerepetition'})).reshape(-1)
    return emg.astype(np.float32), stim.astype(np.int64), rep.astype(np.int64)

def make_windows(emg, labels, reps, win_len=300, step=150):
    n, start, segs = len(labels), 0, []
    while start < n:
        lab, rep, end = int(labels[start]), int(reps[start]), start + 1
        while end < n and int(labels[end]) == lab and int(reps[end]) == rep:
            end += 1
        if lab != 0:
            segs.append((start, end, lab))
        start = end
    X, y = [], []
    for s, e, lab in segs:
        if e - s < win_len:
            continue
        for i in range(s, e - win_len + 1, step):
            X.append(emg[i:i+win_len].T)
            y.append(lab)
    if not X:
        return np.empty((0, emg.shape[1], win_len), np.float32), np.empty((0,), np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y, np.int64)

def subsample(X, y, n, rng):
    idx = []
    for c in np.unique(y):
        i = np.flatnonzero(y == c)
        idx.append(rng.choice(i, min(len(i), n), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]

def common_classes(Xtr, ytr, Xte, yte):
    c = np.intersect1d(np.unique(ytr), np.unique(yte))
    return (Xtr[np.isin(ytr, c)], ytr[np.isin(ytr, c)],
            Xte[np.isin(yte, c)], yte[np.isin(yte, c)], c)

def channel_norm(X):
    m = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    s = X.std(axis=(0, 2),  keepdims=True)
    return m, np.where(s < 1e-6, 1.0, s).astype(np.float32)

def remap(y_tr, y_te):
    cls = np.sort(np.unique(np.concatenate([y_tr, y_te])))
    c2i = {int(c): i for i, c in enumerate(cls)}
    return (np.array([c2i[int(c)] for c in y_tr], np.int64),
            np.array([c2i[int(c)] for c in y_te], np.int64),
            len(cls))


# ── Model ─────────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """Same Conv1D encoder as Latent CNN — output is 128-dim latent."""
    def __init__(self, n_channels, win_len, latent_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        flat = 128 * (win_len // 8)
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.bottleneck(self.encoder(x))


class ConditionalLatentClassifier(nn.Module):
    """
    Conditional Latent CNN.

    What is new vs plain Latent CNN:
      After encoding x to 128-dim latent z, we concatenate a one-hot
      subject vector (length = n_subjects). The combined vector
      [z | subject_onehot] goes through an extra dense layer, then
      the SoftMax classifier.

      This lets the model adjust its decision boundary per-subject,
      factorising subject-specific variation from gesture variation.
    """
    def __init__(self, n_channels, win_len,
                 latent_dim=128, n_subjects=11, n_classes=52,
                 hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder    = ConvEncoder(n_channels, win_len, latent_dim, dropout)
        # combined dim = latent + subject one-hot
        combined_dim    = latent_dim + n_subjects
        self.head       = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.n_subjects = n_subjects

    def forward(self, x, subj_onehot):
        z       = self.encoder(x)                  # (B, latent_dim)
        z_cond  = torch.cat([z, subj_onehot], dim=1)  # (B, latent+n_subj)
        return self.head(z_cond), z                # logits, latent


# ── Train / eval ──────────────────────────────────────────────────────────

def train_model(model, loader, epochs, lr, wd, device):
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(epochs):
        for xb, sb, yb in loader:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb, sb)
            crit(logits, yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  epoch={ep+1:02d}', flush=True)

@torch.no_grad()
def predict_model(model, X, S, batch_size, device):
    model.eval()
    preds, latents = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device)
        sb = torch.from_numpy(S[i:i+batch_size]).to(device)
        logits, z = model(xb, sb)
        preds.append(logits.argmax(1).cpu().numpy())
        latents.append(z.cpu().numpy())
    return (np.concatenate(preds)   if preds   else np.empty((0,), np.int64),
            np.concatenate(latents) if latents else np.empty((0, 128), np.float32))


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',            required=True)
    ap.add_argument('--test',            required=True,  help='e.g. s1')
    ap.add_argument('--out_json',        required=True)
    ap.add_argument('--latent_npz',      default='')
    ap.add_argument('--n_subjects',      type=int,   default=11)
    ap.add_argument('--fs',              type=int,   default=2000)
    ap.add_argument('--win_ms',          type=int,   default=150)
    ap.add_argument('--overlap',         type=float, default=0.5)
    ap.add_argument('--train_per_class', type=int,   default=120)
    ap.add_argument('--test_per_class',  type=int,   default=80)
    ap.add_argument('--seed',            type=int,   default=42)
    ap.add_argument('--epochs',          type=int,   default=50)
    ap.add_argument('--batch_size',      type=int,   default=256)
    ap.add_argument('--lr',              type=float, default=1e-3)
    ap.add_argument('--weight_decay',    type=float, default=1e-4)
    ap.add_argument('--latent_dim',      type=int,   default=128)
    ap.add_argument('--hidden_dim',      type=int,   default=128)
    ap.add_argument('--dropout',         type=float, default=0.3)
    ap.add_argument('--cpus',            type=int,   default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, args.cpus))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base = Path(args.base)
    sds  = sorted([p for p in base.iterdir()
                   if p.is_dir() and p.name.startswith('s')],
                  key=lambda x: int(x.name.replace('s', '')))
    if not sds:
        raise RuntimeError(f'No subject dirs under {base}')

    # Build subject index map  s1→0, s2→1, ...
    subj_names = [p.name for p in sds]
    subj2idx   = {n: i for i, n in enumerate(subj_names)}
    N_SUBJ     = len(sds)

    win_len = int(args.fs * args.win_ms / 1000)   # 300
    step    = max(1, int(win_len * (1 - args.overlap)))  # 150

    # Load all subjects
    Xtr_all, ytr_all, str_all = [], [], []
    Xte_all, yte_all          = [], []

    for sd in sds:
        emg, lbl, rep = load_subject(sd)
        X, y = make_windows(emg, lbl, rep, win_len, step)
        sidx = subj2idx[sd.name]
        if sd.name == args.test:
            Xte_all.append(X)
            yte_all.append(y)
            # test subject one-hot
            S_te = np.zeros((len(X), N_SUBJ), dtype=np.float32)
            S_te[:, sidx] = 1.0
        else:
            Xtr_all.append(X)
            ytr_all.append(y)
            S_tr = np.zeros((len(X), N_SUBJ), dtype=np.float32)
            S_tr[:, sidx] = 1.0
            str_all.append(S_tr)

    if not Xtr_all or not Xte_all:
        raise RuntimeError('Train/test split failed')

    X_train = np.concatenate(Xtr_all)
    y_train = np.concatenate(ytr_all)
    S_train = np.concatenate(str_all)   # subject one-hots for training
    X_test  = np.concatenate(Xte_all)
    y_test  = np.concatenate(yte_all)
    # test one-hot: all same (the held-out subject)
    test_idx = subj2idx[args.test]
    S_test   = np.zeros((len(X_test), N_SUBJ), dtype=np.float32)
    S_test[:, test_idx] = 1.0

    X_train, y_train, X_test, y_test, _ = common_classes(
        X_train, y_train, X_test, y_test)
    # apply same mask to S_train / S_test
    mask_tr = np.isin(np.concatenate(ytr_all),
                      np.intersect1d(np.unique(y_train), np.unique(y_test)))
    # re-filter properly
    common = np.intersect1d(
        np.unique(np.concatenate(ytr_all)),
        np.unique(np.concatenate(yte_all)))
    Xtr2, ytr2, Str2 = [], [], []
    for X, y, S in zip(Xtr_all, ytr_all, str_all):
        m = np.isin(y, common)
        Xtr2.append(X[m]); ytr2.append(y[m]); Str2.append(S[m])
    Xte2, yte2 = [], []
    for X, y in zip(Xte_all, yte_all):
        m = np.isin(y, common)
        Xte2.append(X[m]); yte2.append(y[m])

    X_train = np.concatenate(Xtr2); y_train = np.concatenate(ytr2)
    S_train = np.concatenate(Str2)
    X_test  = np.concatenate(Xte2); y_test  = np.concatenate(yte2)
    S_test  = np.zeros((len(X_test), N_SUBJ), dtype=np.float32)
    S_test[:, test_idx] = 1.0

    std_ratio = float(X_test.std() / (X_train.std() + 1e-12))

    X_train, y_train = subsample(X_train, y_train, args.train_per_class, rng)
    # subsample S_train to match
    rng2 = np.random.default_rng(args.seed)
    idx_s = []
    for c in np.unique(y_train):
        ii = np.flatnonzero(y_train == c)
        idx_s.append(rng2.choice(ii, min(len(ii), args.train_per_class), replace=False))
    idx_s = np.concatenate(idx_s); rng2.shuffle(idx_s)
    X_train = X_train[idx_s]; y_train = y_train[idx_s]
    S_train = S_train[idx_s]

    X_test, y_test = subsample(X_test, y_test, args.test_per_class, rng)
    idx_te = []
    rng3 = np.random.default_rng(args.seed)
    for c in np.unique(y_test):
        ii = np.flatnonzero(y_test == c)
        idx_te.append(rng3.choice(ii, min(len(ii), args.test_per_class), replace=False))
    idx_te = np.concatenate(idx_te); rng3.shuffle(idx_te)
    X_test  = X_test[idx_te];  y_test  = y_test[idx_te]
    S_test  = S_test[idx_te]

    cls_sorted  = np.sort(np.unique(np.concatenate([y_train, y_test])))
    c2i         = {int(c): i for i, c in enumerate(cls_sorted)}
    y_train_idx = np.array([c2i[int(c)] for c in y_train], np.int64)
    y_test_idx  = np.array([c2i[int(c)] for c in y_test],  np.int64)
    K = len(cls_sorted)

    m, s    = channel_norm(X_train)
    X_train = (X_train - m) / s
    X_test  = (X_test  - m) / s

    N, C, T = X_train.shape
    print(f'{args.test} | K={K} std_ratio={std_ratio:.3f} '
          f'train={N} test={len(X_test)} device={device}', flush=True)

    ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(S_train),
        torch.from_numpy(y_train_idx))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ConditionalLatentClassifier(
        n_channels=C, win_len=T,
        latent_dim=args.latent_dim,
        n_subjects=N_SUBJ,
        n_classes=K,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Params: {n_params:,}', flush=True)

    train_model(model, loader, args.epochs, args.lr, args.weight_decay, device)

    preds, latent_vecs = predict_model(
        model, X_test, S_test, args.batch_size, device)

    acc      = float(accuracy_score(y_test_idx, preds))
    bacc     = float(balanced_accuracy_score(y_test_idx, preds))
    macro_f1 = float(f1_score(y_test_idx, preds, average='macro', zero_division=0))

    print(f'{args.test} | acc={acc:.4f} bacc={bacc:.4f} f1={macro_f1:.4f}', flush=True)

    if args.latent_npz:
        Path(args.latent_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.latent_npz,
            latents=latent_vecs.astype(np.float32),
            labels=y_test_idx.astype(np.int32),
            gesture_ids=y_test.astype(np.int32),
            subject=np.array([int(args.test.replace('s', ''))]),
        )
        print(f'  Latents saved: {args.latent_npz}', flush=True)

    out = {
        'method':        'conditional_latent_cnn',
        'test_subject':  args.test,
        'N_train':       int(N),
        'N_test':        int(len(X_test)),
        'K':             int(K),
        'n_subjects':    int(N_SUBJ),
        'win_ms':        int(args.win_ms),
        'overlap':       float(args.overlap),
        'strict_purity': True,
        'train_per_class': int(args.train_per_class),
        'test_per_class':  int(args.test_per_class),
        'std_ratio':     std_ratio,
        'acc':           acc,
        'balanced_acc':  bacc,
        'macro_f1':      macro_f1,
        'epochs':        int(args.epochs),
        'latent_dim':    int(args.latent_dim),
        'hidden_dim':    int(args.hidden_dim),
        'dropout':       float(args.dropout),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved: {args.out_json}', flush=True)

if __name__ == '__main__':
    main()
>>>>>>> 9bbfa3b24bb49262e519b5672b0b6636e2cc4682
