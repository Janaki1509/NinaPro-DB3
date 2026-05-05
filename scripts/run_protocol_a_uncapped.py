#!/usr/bin/env python3
"""
run_protocol_a_uncapped.py
===========================
Protocol A comparison — capped vs uncapped training data.

Runs MLP SoftMax under:
  - Protocol A CAPPED:   120 train / 80 test windows per class (matches LOSO fairness)
  - Protocol A UNCAPPED: all available windows, 80/20 random split (matches published papers)

This directly shows whether our lower Protocol A accuracy is due to
the subsampling cap, and whether uncapped results approach Niu et al. (93.82%)
and Sandoval-Espino et al. (90.23%).

Run on laptop:
  cd C:\\ninapro_db3
  python scripts\\run_protocol_a_uncapped.py --base subjects --out_dir outputs\\plots
"""

import argparse, os
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Data loading ──────────────────────────────────────────────────────────

def _maybe_array(x):
    try:
        arr = np.asarray(x)
        if arr.size > 0: return arr
    except: pass
    return None

def _search(obj, keys, depth=0, max_depth=4):
    if depth > max_depth: return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                a = _maybe_array(v)
                if a is not None: return a
        for _, v in obj.items():
            f = _search(v, keys, depth+1, max_depth)
            if f is not None: return f
    if hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            if k in keys:
                a = _maybe_array(v)
                if a is not None: return a
        for _, v in obj.__dict__.items():
            f = _search(v, keys, depth+1, max_depth)
            if f is not None: return f
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            f = _search(item, keys, depth+1, max_depth)
            if f is not None: return f
    return None

def load_subject(sd):
    mats = sorted(Path(sd).rglob('*.mat.mat')) or sorted(Path(sd).rglob('*.mat'))
    if not mats: raise FileNotFoundError(f'No .mat files in {sd}')
    data = sio.loadmat(str(mats[0]), squeeze_me=True, struct_as_record=False)
    emg  = np.asarray(_search(data, {'emg'}))
    stim = np.asarray(_search(data, {'restimulus'})).reshape(-1)
    rep  = np.asarray(_search(data, {'rerepetition'})).reshape(-1)
    return emg.astype(np.float32), stim.astype(np.int64), rep.astype(np.int64)

def make_windows(emg, labels, reps, win_len=300, step=150):
    n, start, segs = len(labels), 0, []
    while start < n:
        lab, rep, end = int(labels[start]), int(reps[start]), start+1
        while end < n and int(labels[end])==lab and int(reps[end])==rep: end+=1
        if lab != 0: segs.append((start, end, lab))
        start = end
    X, y = [], []
    for s, e, lab in segs:
        if e-s < win_len: continue
        for i in range(s, e-win_len+1, step):
            X.append(emg[i:i+win_len].T); y.append(lab)
    if not X:
        return np.empty((0,emg.shape[1],win_len),np.float32), np.empty((0,),np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y,np.int64)

def channel_norm(X):
    m = X.mean(axis=(0,2),keepdims=True).astype(np.float32)
    s = X.std(axis=(0,2),keepdims=True)
    return m, np.where(s<1e-6,1.0,s).astype(np.float32)

def remap(y_tr, y_te):
    cls = np.sort(np.unique(np.concatenate([y_tr, y_te])))
    c2i = {int(c):i for i,c in enumerate(cls)}
    return (np.array([c2i[int(c)] for c in y_tr],np.int64),
            np.array([c2i[int(c)] for c in y_te],np.int64),
            len(cls))

def subsample(X, y, n, rng):
    idx = []
    for c in np.unique(y):
        i = np.flatnonzero(y==c)
        idx.append(rng.choice(i, min(len(i),n), replace=False))
    idx = np.concatenate(idx); rng.shuffle(idx)
    return X[idx], y[idx]


# ── MLP ───────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,K))
    def forward(self, x): return self.net(x)

def train_eval(X_tr, y_tr, X_te, y_te, epochs=30):
    """Train MLP and return balanced accuracy and total accuracy."""
    common = np.intersect1d(np.unique(y_tr), np.unique(y_te))
    X_tr = X_tr[np.isin(y_tr,common)]; y_tr = y_tr[np.isin(y_tr,common)]
    X_te = X_te[np.isin(y_te,common)]; y_te = y_te[np.isin(y_te,common)]
    if len(common)==0: return None, None, None, None

    m, s = channel_norm(X_tr)
    X_tr = (X_tr-m)/s; X_te = (X_te-m)/s
    N,C,T = X_tr.shape; K = len(common)
    y_tr_i, y_te_i, _ = remap(y_tr, y_te)

    Xf_tr = torch.from_numpy(X_tr.reshape(N,C*T))
    Xf_te = torch.from_numpy(X_te.reshape(len(X_te),C*T))

    model = MLP(C*T, K)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xf_tr, torch.from_numpy(y_tr_i)),
                        batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb),yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(Xf_te).argmax(1).numpy()
    return (float(balanced_accuracy_score(y_te_i, preds)),
            float(accuracy_score(y_te_i, preds)),
            int(N), int(len(X_te)))


# ── Main experiment ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',    default='subjects')
    ap.add_argument('--out_dir', default='outputs/plots')
    ap.add_argument('--epochs',  type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = Path(args.base)
    sds  = sorted([p for p in base.iterdir()
                   if p.is_dir() and p.name.startswith('s')],
                  key=lambda x: int(x.name.replace('s','')))

    print(f"Running Protocol A (capped vs uncapped) on {len(sds)} subjects...")
    print(f"Epochs per subject: {args.epochs}")

    capped_results   = []
    uncapped_results = []

    for sd in sds:
        print(f"\n  Subject {sd.name}...")
        emg, lbl, rep = load_subject(sd)
        X, y = make_windows(emg, lbl, rep)
        if len(X) == 0:
            print(f"    No windows found, skipping")
            continue

        rng = np.random.default_rng(42)
        idx = np.arange(len(X)); rng.shuffle(idx)
        split = int(0.8*len(idx))
        tr, te = idx[:split], idx[split:]

        # CAPPED: 120/80 per class (matches LOSO paper fairness)
        rng2 = np.random.default_rng(42)
        Xc_tr, yc_tr = subsample(X[tr], y[tr], 120, rng2)
        rng3 = np.random.default_rng(42)
        Xc_te, yc_te = subsample(X[te], y[te], 80, rng3)
        bacc_c, acc_c, n_tr_c, n_te_c = train_eval(Xc_tr, yc_tr, Xc_te, yc_te, args.epochs)

        # UNCAPPED: all windows (matches published papers like Niu et al.)
        bacc_u, acc_u, n_tr_u, n_te_u = train_eval(X[tr], y[tr], X[te], y[te], args.epochs)

        if bacc_c is not None:
            print(f"    CAPPED   (120/80): bacc={bacc_c:.4f} acc={acc_c:.4f} "
                  f"train={n_tr_c} test={n_te_c}")
            capped_results.append({'subject':sd.name,'bacc':bacc_c,'acc':acc_c,
                                   'n_train':n_tr_c,'n_test':n_te_c})

        if bacc_u is not None:
            print(f"    UNCAPPED (all):    bacc={bacc_u:.4f} acc={acc_u:.4f} "
                  f"train={n_tr_u} test={n_te_u}")
            uncapped_results.append({'subject':sd.name,'bacc':bacc_u,'acc':acc_u,
                                     'n_train':n_tr_u,'n_test':n_te_u})

    # Summary
    if capped_results and uncapped_results:
        mean_c_bacc = np.mean([r['bacc'] for r in capped_results])
        mean_u_bacc = np.mean([r['bacc'] for r in uncapped_results])
        mean_c_acc  = np.mean([r['acc']  for r in capped_results])
        mean_u_acc  = np.mean([r['acc']  for r in uncapped_results])

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Protocol A CAPPED   (120/80 per class): bacc={mean_c_bacc:.4f} "
              f"acc={mean_c_acc:.4f}")
        print(f"Protocol A UNCAPPED (all windows):      bacc={mean_u_bacc:.4f} "
              f"acc={mean_u_acc:.4f}")
        print(f"Protocol B LOSO (confirmed):            bacc=0.0593")
        print(f"\nPublished baselines (within-subject):")
        print(f"  Niu et al. 2024 (DenseNet, 8 subjects):       93.82%")
        print(f"  Sandoval-Espino 2022 (CNN, 150ms):             90.23%")
        print(f"\nKey finding:")
        print(f"  Capped→Uncapped gap:  {(mean_u_acc - mean_c_acc)*100:.1f}% accuracy")
        print(f"  Uncapped→Published gap: "
              f"{(0.9023 - mean_u_acc)*100:.1f}% remaining difference")
        print(f"  Uncapped→LOSO gap:    {(mean_u_acc - 0.0593)*100:.1f}%")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        # Left: bar comparison
        ax = axes[0]
        methods = ['Protocol A\nCAPPED\n(120/80 per class)',
                   'Protocol A\nUNCAPPED\n(all windows)',
                   'Protocol B\nLOSO\n(cross-subject)',
                   'Niu et al.\n2024\n(DenseNet)',
                   'Sandoval\n2022\n(CNN)']
        values  = [mean_c_acc, mean_u_acc, 0.0593, 0.9382, 0.9023]
        colors  = ['#2980b9','#1a5276','#e74c3c','#95a5a6','#bdc3c7']
        bars = ax.bar(methods, values, color=colors, edgecolor='white',
                      linewidth=1.5, zorder=3, width=0.55)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                    f'{val:.1%}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Mean Accuracy', fontsize=11)
        ax.set_title('Within-subject vs cross-subject accuracy\n'
                     'Protocol A capped vs uncapped vs published results',
                     fontsize=10, fontweight='bold')
        ax.axhline(0.15, color='#888', linestyle='--', lw=1.2,
                   label='LOSO target (0.15)', alpha=0.7)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, zorder=0); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8.5)

        # Right: per-subject capped vs uncapped
        ax2 = axes[1]
        subjects = [r['subject'] for r in capped_results]
        x = np.arange(len(subjects)); w = 0.35
        ax2.bar(x-w/2, [r['acc'] for r in capped_results], w,
                label='Protocol A capped (120/80)', color='#2980b9',
                edgecolor='white', zorder=3)
        ax2.bar(x+w/2, [r['acc'] for r in uncapped_results], w,
                label='Protocol A uncapped (all)', color='#1a5276',
                edgecolor='white', zorder=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels([s.upper() for s in subjects], fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('Per-subject: capped vs uncapped Protocol A\n'
                      'Removing the cap shows the true within-subject ceiling',
                      fontsize=10, fontweight='bold')
        ax2.legend(fontsize=9, framealpha=0.9)
        ax2.grid(axis='y', alpha=0.3, zorder=0); ax2.set_axisbelow(True)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

        plt.suptitle('Effect of training data cap on Protocol A accuracy\n'
                     'Cap was intentional to ensure fair comparison with LOSO benchmark',
                     fontsize=11, fontweight='bold', y=1.02)
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, 'protocol_a_capped_vs_uncapped.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: {out_path}")

if __name__ == '__main__':
    main()
