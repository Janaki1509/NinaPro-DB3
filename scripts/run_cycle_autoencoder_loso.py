#!/usr/bin/env python3
"""
run_cycle_autoencoder_loso.py
Cycle-Consistent Cross-Subject Autoencoder for NinaPro DB3 LOSO.
"""

import argparse
import json
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

# Prevent running on login node
if socket.gethostname().startswith("port"):
    print("ERROR: on login node. Submit via sbatch only.", flush=True)
    sys.exit(1)


# ---------------- DATA ---------------- #

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
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            f = _search(item, keys, depth+1, max_depth)
            if f is not None:
                return f
    return None

def load_subject(sd):
    mats = sorted(Path(sd).rglob('*.mat.mat')) or sorted(Path(sd).rglob('*.mat'))
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

    return np.stack(X).astype(np.float32), np.asarray(y, np.int64)


# ---------------- MODEL ---------------- #

class SharedEncoder(nn.Module):
    def __init__(self, n_channels, win_len, latent_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Linear(128 * (win_len // 8), latent_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


class SubjectDecoder(nn.Module):
    def __init__(self, latent_dim, n_channels, win_len):
        super().__init__()
        self.win_len = win_len
        flat = 128 * (win_len // 8)

        self.fc = nn.Linear(latent_dim, flat)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, n_channels, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, z):
        B = z.size(0)
        h = self.fc(z).view(B, 128, self.win_len // 8)
        out = self.deconv(h)

        # FIX: enforce exact length
        if out.shape[-1] != self.win_len:
            out = F.interpolate(out, size=self.win_len, mode='linear', align_corners=False)

        return out


class CycleAutoencoder(nn.Module):
    def __init__(self, n_channels, win_len, latent_dim, n_subjects, n_gestures):
        super().__init__()
        self.encoder = SharedEncoder(n_channels, win_len, latent_dim)
        self.decoders = nn.ModuleList([
            SubjectDecoder(latent_dim, n_channels, win_len)
            for _ in range(n_subjects)
        ])
        self.classifier = nn.Linear(latent_dim, n_gestures)
        self.n_subjects = n_subjects

    def forward(self, x, sidx):
        z = self.encoder(x)
        x_recon = self.decoders[sidx](z)
        logits = self.classifier(z)
        return logits, x_recon, z


# ---------------- TRAIN ---------------- #

def train_model(model, loaders, epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        for loader, sidx in loaders:
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()

                logits, x_recon, z = model(xb, sidx)

                # FIX: match sizes before loss
                if x_recon.shape[-1] != xb.shape[-1]:
                    x_recon = F.interpolate(x_recon, size=xb.shape[-1], mode='linear', align_corners=False)

                loss = crit(logits, yb) + F.mse_loss(x_recon, xb)
                loss.backward()
                opt.step()

        print(f"Epoch {ep+1} done")


# ---------------- MAIN ---------------- #

def main():
    args = argparse.Namespace(
        base="~/ninapro_db3/data",
        test="s1",
        epochs=5
    )

    base = Path(args.base).expanduser()
    sds = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith('s')])

    subj2idx = {p.name: i for i, p in enumerate(sds)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_list, ytr_list, loaders = [], [], []

    for sd in sds:
        emg, lbl, rep = load_subject(sd)
        X, y = make_windows(emg, lbl, rep)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=128, shuffle=True)

        loaders.append((loader, subj2idx[sd.name]))

    model = CycleAutoencoder(
        n_channels=X.shape[1],
        win_len=300,
        latent_dim=128,
        n_subjects=len(sds),
        n_gestures=17
    ).to(device)

    train_model(model, loaders, args.epochs, device)


if __name__ == "__main__":
    main()