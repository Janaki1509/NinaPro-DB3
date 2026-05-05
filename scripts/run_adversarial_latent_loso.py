#!/usr/bin/env python3
"""
run_adversarial_latent_loso.py
================================
Domain Adversarial Latent CNN for NinaPro DB3 LOSO benchmark.

Professor Action Item 3: Domain Adaptation via Adversarial Domain Classifier.

What is new vs conditional_latent_loso.py:
  - Gradient Reversal Layer (GRL) between encoder and subject classifier
  - Encoder is trained to:
      (1) maximize gesture classification accuracy
      (2) simultaneously FOOL the subject classifier (via GRL)
  - This forces the encoder to produce subject-invariant representations
  - The t-SNE should show gesture clusters mixed across subject colors

Architecture:
  Input (12,300)
      -> Conv1D Encoder -> 128-dim latent z
      -> Gesture classifier head  (cross-entropy loss, minimize)
      -> Subject classifier head  (cross-entropy loss, MAXIMIZE via GRL)

The GRL flips the gradient sign during backprop, so the encoder
learns to remove subject information from z.
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
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

if socket.gethostname().startswith("port"):
    print("ERROR: on login node. Submit via sbatch only.", flush=True)
    sys.exit(1)


# ── Gradient Reversal Layer ───────────────────────────────────────────────

class GradientReversalFunction(Function):
    """
    Forward pass: identity (passes input through unchanged)
    Backward pass: multiplies gradient by -lambda (reverses and scales it)
    This makes the encoder try to FOOL the subject classifier.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.saved_tensors[0]
        return -lambda_ * grad_output, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ── Data loading (identical to all other scripts) ─────────────────────────

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
        raise FileNotFoundError(f'No .mat in {sd}')
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
    idx = np.concatenate(idx); rng.shuffle(idx)
    return X[idx], y[idx]

def common_classes(Xtr, ytr, Xte, yte):
    c = np.intersect1d(np.unique(ytr), np.unique(yte))
    return (Xtr[np.isin(ytr, c)], ytr[np.isin(ytr, c)],
            Xte[np.isin(yte, c)], yte[np.isin(yte, c)], c)

def channel_norm(X):
    m = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    s = X.std(axis=(0, 2), keepdims=True)
    return m, np.where(s < 1e-6, 1.0, s).astype(np.float32)


# ── Model ─────────────────────────────────────────────────────────────────

class AdversarialLatentClassifier(nn.Module):
    """
    Domain Adversarial CNN.

    Encoder -> latent z (128-dim)
        -> gesture_head(z)         # trained normally
        -> subject_head(GRL(z))    # encoder trained to fool this
    """
    def __init__(self, n_channels, win_len,
                 latent_dim=128, n_gestures=17, n_subjects=11,
                 dropout=0.3, lambda_=1.0):
        super().__init__()

        # Shared Conv1D encoder
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

        # Gesture classifier (trained to minimize gesture loss)
        self.gesture_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_gestures),
        )

        # Subject classifier with GRL
        # Encoder is trained to FOOL this via gradient reversal
        self.grl = GradientReversal(lambda_=lambda_)
        self.subject_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_subjects),
        )

    def forward(self, x):
        z = self.bottleneck(self.encoder(x))
        gesture_logits = self.gesture_head(z)
        subject_logits = self.subject_head(self.grl(z))
        return gesture_logits, subject_logits, z

    def set_lambda(self, lambda_):
        """Increase lambda_ over training for stable learning."""
        self.grl.lambda_ = lambda_


# ── Training ──────────────────────────────────────────────────────────────

def train_model(model, loader, epochs, lr, wd, device, alpha=0.5):
    """
    alpha: weight of domain adversarial loss
           total_loss = gesture_loss + alpha * domain_loss
    lambda_ increases from 0 to 1 over training (schedule from DANN paper)
    """
    gesture_crit = nn.CrossEntropyLoss()
    subject_crit = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    model.train()
    for ep in range(epochs):
        # Gradually increase lambda_ (standard DANN schedule)
        p = ep / epochs
        lambda_ = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        model.set_lambda(lambda_)

        total_g, total_d, n_seen = 0.0, 0.0, 0
        for xb, yb_g, yb_s in loader:
            xb   = xb.to(device)
            yb_g = yb_g.to(device)
            yb_s = yb_s.to(device)

            opt.zero_grad()
            g_logits, s_logits, _ = model(xb)

            g_loss = gesture_crit(g_logits, yb_g)
            d_loss = subject_crit(s_logits, yb_s)
            loss   = g_loss + alpha * d_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_g += g_loss.item() * len(yb_g)
            total_d += d_loss.item() * len(yb_g)
            n_seen  += len(yb_g)

        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  epoch={ep+1:02d} '
                  f'g_loss={total_g/n_seen:.4f} '
                  f'd_loss={total_d/n_seen:.4f} '
                  f'lambda={lambda_:.3f}', flush=True)

@torch.no_grad()
def predict_model(model, X, batch_size, device):
    model.eval()
    preds, latents = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device)
        g_logits, _, z = model(xb)
        preds.append(g_logits.argmax(1).cpu().numpy())
        latents.append(z.cpu().numpy())
    return (np.concatenate(preds)   if preds   else np.empty((0,), np.int64),
            np.concatenate(latents) if latents else np.empty((0, 128), np.float32))


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',            required=True)
    ap.add_argument('--test',            required=True)
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
    ap.add_argument('--alpha',           type=float, default=0.5,
                    help='Weight of domain adversarial loss')
    ap.add_argument('--dropout',         type=float, default=0.3)
    ap.add_argument('--cpus',            type=int,   default=4)
    return ap.parse_args()


def main():
    args   = parse_args()
    rng    = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, args.cpus))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base = Path(args.base)
    sds  = sorted([p for p in base.iterdir()
                   if p.is_dir() and p.name.startswith('s')],
                  key=lambda x: int(x.name.replace('s', '')))
    if not sds:
        raise RuntimeError(f'No subject dirs under {base}')

    subj2idx = {p.name: i for i, p in enumerate(sds)}
    N_SUBJ   = len(sds)
    win_len  = int(args.fs * args.win_ms / 1000)
    step     = max(1, int(win_len * (1 - args.overlap)))

    Xtr_all, ytr_all, Str_all = [], [], []
    Xte_all, yte_all          = [], []

    for sd in sds:
        emg, lbl, rep = load_subject(sd)
        X, y = make_windows(emg, lbl, rep, win_len, step)
        sidx = subj2idx[sd.name]
        if sd.name == args.test:
            Xte_all.append(X); yte_all.append(y)
        else:
            Xtr_all.append(X); ytr_all.append(y)
            s_labels = np.full(len(y), sidx, dtype=np.int64)
            Str_all.append(s_labels)

    X_train = np.concatenate(Xtr_all)
    y_train = np.concatenate(ytr_all)
    S_train = np.concatenate(Str_all)   # subject labels for domain loss
    X_test  = np.concatenate(Xte_all)
    y_test  = np.concatenate(yte_all)

    # Common classes
    common = np.intersect1d(np.unique(y_train), np.unique(y_test))
    m_tr = np.isin(y_train, common); m_te = np.isin(y_test, common)
    X_train, y_train, S_train = X_train[m_tr], y_train[m_tr], S_train[m_tr]
    X_test,  y_test            = X_test[m_te],  y_test[m_te]

    std_ratio = float(X_test.std() / (X_train.std() + 1e-12))

    # Subsample
    X_train, y_train = subsample(X_train, y_train, args.train_per_class, rng)
    # match S_train to subsampled indices
    rng2 = np.random.default_rng(args.seed)
    idx_s = []
    for c in np.unique(y_train):
        ii = np.flatnonzero(y_train == c)
        idx_s.append(rng2.choice(ii, min(len(ii), args.train_per_class), replace=False))
    idx_s = np.concatenate(idx_s); rng2.shuffle(idx_s)
    X_train = X_train[idx_s]; y_train = y_train[idx_s]; S_train = S_train[idx_s]

    X_test, y_test = subsample(X_test, y_test, args.test_per_class, rng)

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
        torch.from_numpy(y_train_idx),
        torch.from_numpy(S_train))
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)

    model = AdversarialLatentClassifier(
        n_channels=C, win_len=T,
        latent_dim=args.latent_dim,
        n_gestures=K,
        n_subjects=N_SUBJ,
        dropout=args.dropout,
        lambda_=0.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Params: {n_params:,}  alpha={args.alpha}', flush=True)

    train_model(model, loader, args.epochs,
                args.lr, args.weight_decay, device, args.alpha)

    preds, latent_vecs = predict_model(
        model, X_test, args.batch_size, device)

    acc      = float(accuracy_score(y_test_idx, preds))
    bacc     = float(balanced_accuracy_score(y_test_idx, preds))
    macro_f1 = float(f1_score(y_test_idx, preds,
                               average='macro', zero_division=0))

    print(f'{args.test} | acc={acc:.4f} bacc={bacc:.4f} f1={macro_f1:.4f}',
          flush=True)

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
        'method':          'adversarial_latent_cnn',
        'test_subject':    args.test,
        'N_train':         int(N),
        'N_test':          int(len(X_test)),
        'K':               int(K),
        'n_subjects':      int(N_SUBJ),
        'win_ms':          int(args.win_ms),
        'overlap':         float(args.overlap),
        'strict_purity':   True,
        'train_per_class': int(args.train_per_class),
        'test_per_class':  int(args.test_per_class),
        'std_ratio':       std_ratio,
        'acc':             acc,
        'balanced_acc':    bacc,
        'macro_f1':        macro_f1,
        'epochs':          int(args.epochs),
        'latent_dim':      int(args.latent_dim),
        'alpha':           float(args.alpha),
        'dropout':         float(args.dropout),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved: {args.out_json}', flush=True)

if __name__ == '__main__':
    main()
