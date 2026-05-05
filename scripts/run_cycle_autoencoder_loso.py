#!/usr/bin/env python3
"""
run_cycle_autoencoder_loso.py
==============================
Cycle-Consistent Cross-Subject Autoencoder for NinaPro DB3 LOSO.

Professor Item 4: "High Innovation" — Cross-Domain Translation.

Architecture:
  - Shared encoder E: (C,T) -> 128-dim shared latent z
  - Subject-specific decoders D_i: z -> (C,T) for subject i
  - Gesture classifier G: z -> K classes
  - Cycle consistency: E(D_i(E(x))) ≈ E(x)

Training losses:
  1. Gesture classification loss   (learn gesture features)
  2. Reconstruction loss           (decoder can recreate input)
  3. Cycle consistency loss        (latent stable after decode+encode)
  4. Cross-subject translation     (decode to another subject, re-encode)

The cycle consistency forces the encoder to produce representations
that are invariant to WHICH subject decoded the signal — the key
property needed for cross-subject generalization.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if socket.gethostname().startswith("port"):
    print("ERROR: on login node. Submit via sbatch only.", flush=True)
    sys.exit(1)


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

def channel_norm(X):
    m = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    s = X.std(axis=(0, 2), keepdims=True)
    return m, np.where(s < 1e-6, 1.0, s).astype(np.float32)


# ── Model ─────────────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    """Shared Conv1D encoder — same as Latent CNN."""
    def __init__(self, n_channels, win_len, latent_dim=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.flat_size = 128 * (win_len // 8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class SubjectDecoder(nn.Module):
    """
    Subject-specific decoder: z -> (C, T).
    Each subject has its own decoder that learns that subject's
    EMG signal style. The shared encoder must produce z that
    works across ALL decoders — forcing subject invariance.
    """
    def __init__(self, latent_dim, n_channels, win_len):
        super().__init__()
        self.win_len     = win_len
        self.n_channels  = n_channels
        flat_size        = 128 * (win_len // 8)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, flat_size),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, n_channels, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, z):
        B = z.size(0)
        h = self.fc(z).view(B, 128, self.win_len // 8)
        return self.deconv(h)[:, :, :self.win_len]


class CycleAutoencoder(nn.Module):
    """
    Full cycle-consistent cross-subject model.

    Components:
      encoder:   shared Conv1D encoder
      decoders:  one per training subject (subject-specific)
      classifier: gesture classifier on shared latent
    """
    def __init__(self, n_channels, win_len, latent_dim=128,
                 n_subjects=11, n_gestures=17, dropout=0.3):
        super().__init__()
        self.encoder    = SharedEncoder(n_channels, win_len, latent_dim, dropout)
        self.decoders   = nn.ModuleList([
            SubjectDecoder(latent_dim, n_channels, win_len)
            for _ in range(n_subjects)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_gestures),
        )
        self.n_subjects = n_subjects

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, subj_idx):
        return self.decoders[subj_idx](z)

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x, subj_idx):
        z        = self.encode(x)
        x_recon  = self.decode(z, subj_idx)
        g_logits = self.classify(z)
        return g_logits, x_recon, z


# ── Training ──────────────────────────────────────────────────────────────

def train_model(model, X_all, y_all, S_all, epochs, lr, wd, device,
                batch_size, lambda_recon=1.0, lambda_cycle=0.5):
    """
    X_all: list of arrays (one per training subject)
    y_all: list of label arrays
    S_all: list of subject indices

    Losses:
      gesture_loss:  cross-entropy on gesture predictions
      recon_loss:    MSE between input and reconstructed signal
      cycle_loss:    MSE between original latent and re-encoded latent
                     after cross-subject translation
    """
    gesture_crit = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Build per-subject dataloaders
    loaders = []
    for X, y, sidx in zip(X_all, y_all, S_all):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loaders.append((DataLoader(ds, batch_size=batch_size,
                                   shuffle=True, num_workers=0), sidx))

    model.train()
    for ep in range(epochs):
        total_g, total_r, total_c, n_seen = 0., 0., 0., 0

        # Iterate through all subjects each epoch
        for loader, sidx in loaders:
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()

                # Forward through own subject
                g_logits, x_recon, z = model(xb, sidx)

                # Gesture loss
                g_loss = gesture_crit(g_logits, yb)

                # Reconstruction loss: can the decoder recreate the input?
                r_loss = F.mse_loss(x_recon, xb)

                # Cycle consistency: translate to a random other subject,
                # re-encode, check that latent is similar
                other_sidx = (sidx + 1) % model.n_subjects
                with torch.no_grad():
                    x_translated = model.decode(z.detach(), other_sidx)
                z_cycle = model.encode(x_translated)
                c_loss  = F.mse_loss(z_cycle, z.detach())

                loss = g_loss + lambda_recon * r_loss + lambda_cycle * c_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                total_g += g_loss.item() * len(yb)
                total_r += r_loss.item() * len(yb)
                total_c += c_loss.item() * len(yb)
                n_seen  += len(yb)

        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  epoch={ep+1:02d} '
                  f'g={total_g/n_seen:.4f} '
                  f'recon={total_r/n_seen:.4f} '
                  f'cycle={total_c/n_seen:.4f}', flush=True)


@torch.no_grad()
def predict_model(model, X, sidx, batch_size, device):
    model.eval()
    preds, latents = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device)
        g_logits, _, z = model(xb, sidx)
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
    ap.add_argument('--fs',              type=int,   default=2000)
    ap.add_argument('--win_ms',          type=int,   default=150)
    ap.add_argument('--overlap',         type=float, default=0.5)
    ap.add_argument('--train_per_class', type=int,   default=120)
    ap.add_argument('--test_per_class',  type=int,   default=80)
    ap.add_argument('--seed',            type=int,   default=42)
    ap.add_argument('--epochs',          type=int,   default=50)
    ap.add_argument('--batch_size',      type=int,   default=128)
    ap.add_argument('--lr',              type=float, default=5e-4)
    ap.add_argument('--weight_decay',    type=float, default=1e-4)
    ap.add_argument('--latent_dim',      type=int,   default=128)
    ap.add_argument('--lambda_recon',    type=float, default=1.0)
    ap.add_argument('--lambda_cycle',    type=float, default=0.5)
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

    # Load all subjects, keep per-subject arrays for cycle training
    Xtr_list, ytr_list, Str_list = [], [], []
    X_test,   y_test             = None, None
    test_sidx = subj2idx[args.test]

    for sd in sds:
        emg, lbl, rep = load_subject(sd)
        X, y = make_windows(emg, lbl, rep, win_len, step)
        sidx = subj2idx[sd.name]
        if sd.name == args.test:
            X_test, y_test = X, y
        else:
            Xtr_list.append(X)
            ytr_list.append(y)
            Str_list.append(sidx)

    # Common classes across all subjects
    all_labels = np.concatenate(ytr_list + [y_test])
    common = np.intersect1d(
        np.unique(np.concatenate(ytr_list)),
        np.unique(y_test))

    # Filter each training subject to common classes
    Xtr_filt, ytr_filt = [], []
    for X, y in zip(Xtr_list, ytr_list):
        m = np.isin(y, common)
        Xtr_filt.append(X[m]); ytr_filt.append(y[m])

    m_te = np.isin(y_test, common)
    X_test, y_test = X_test[m_te], y_test[m_te]

    # Remap labels
    cls_sorted = np.sort(common)
    c2i = {int(c): i for i, c in enumerate(cls_sorted)}
    K   = len(cls_sorted)

    # Subsample per-subject train sets
    Xtr_sub, ytr_sub = [], []
    for X, y in zip(Xtr_filt, ytr_filt):
        Xs, ys = subsample(X, y, args.train_per_class, rng)
        Xtr_sub.append(Xs); ytr_sub.append(ys)

    X_test, y_test = subsample(X_test, y_test, args.test_per_class, rng)

    # Compute channel norm from all training data combined
    X_train_all = np.concatenate(Xtr_sub)
    y_train_all = np.concatenate(ytr_sub)
    std_ratio = float(X_test.std() / (X_train_all.std() + 1e-12))

    m, s = channel_norm(X_train_all)
    Xtr_norm = [(X - m) / s for X in Xtr_sub]
    X_test_n = (X_test - m) / s

    # Remap labels to 0..K-1
    ytr_idx = [np.array([c2i[int(c)] for c in y], np.int64) for y in ytr_sub]
    yte_idx  = np.array([c2i[int(c)] for c in y_test], np.int64)

    N_train = sum(len(X) for X in Xtr_norm)
    print(f'{args.test} | K={K} std_ratio={std_ratio:.3f} '
          f'train={N_train} test={len(X_test)} device={device}', flush=True)

    model = CycleAutoencoder(
        n_channels=Xtr_norm[0].shape[1],
        win_len=win_len,
        latent_dim=args.latent_dim,
        n_subjects=N_SUBJ,
        n_gestures=K,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Params: {n_params:,}  '
          f'lambda_recon={args.lambda_recon}  '
          f'lambda_cycle={args.lambda_cycle}', flush=True)

    train_model(
        model, Xtr_norm, ytr_idx, Str_list,
        epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
        device=device, batch_size=args.batch_size,
        lambda_recon=args.lambda_recon,
        lambda_cycle=args.lambda_cycle,
    )

    # At test time, use the test subject's decoder index
    # (decoder for test subject was not trained — use nearest trained subject)
    # We use the encoder directly since it is shared
    preds, latent_vecs = predict_model(
        model, X_test_n, test_sidx % (N_SUBJ - 1),
        args.batch_size, device)

    acc      = float(accuracy_score(yte_idx, preds))
    bacc     = float(balanced_accuracy_score(yte_idx, preds))
    macro_f1 = float(f1_score(yte_idx, preds, average='macro', zero_division=0))

    print(f'{args.test} | acc={acc:.4f} bacc={bacc:.4f} f1={macro_f1:.4f}',
          flush=True)

    if args.latent_npz:
        Path(args.latent_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.latent_npz,
            latents=latent_vecs.astype(np.float32),
            labels=yte_idx.astype(np.int32),
            gesture_ids=y_test.astype(np.int32),
            subject=np.array([int(args.test.replace('s', ''))]),
        )
        print(f'  Latents saved: {args.latent_npz}', flush=True)

    out = {
        'method':          'cycle_autoencoder',
        'test_subject':    args.test,
        'N_train':         int(N_train),
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
        'lambda_recon':    float(args.lambda_recon),
        'lambda_cycle':    float(args.lambda_cycle),
        'dropout':         float(args.dropout),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  Saved: {args.out_json}', flush=True)

if __name__ == '__main__':
    main()
