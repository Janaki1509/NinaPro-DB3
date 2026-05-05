#!/usr/bin/env python3
"""
compute_mmd_domain_shift.py
============================
Replaces std_ratio with proper Maximum Mean Discrepancy (MMD)
for the domain shift vs LOSO accuracy figure.

MMD measures the distance between two distributions in a
reproducing kernel Hilbert space. It is the canonical metric
for domain shift / distribution mismatch in machine learning.

MMD^2(X_s, X_t) = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
where k is the RBF kernel and x~X_s, y~X_t.

Higher MMD = more domain shift between subjects = harder LOSO.

Usage (on laptop):
  cd C:\\ninapro_db3
  python scripts\\compute_mmd_domain_shift.py --base subjects --out_dir outputs\\plots
"""

import argparse
import os
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
    if not mats: raise FileNotFoundError(f'No .mat in {sd}')
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


# ── MMD computation ───────────────────────────────────────────────────────

def rbf_kernel(X, Y, sigma=None):
    """
    RBF kernel matrix between rows of X and Y.
    k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
    If sigma is None, uses the median heuristic (standard practice).
    """
    # Flatten windows to vectors: (N, C*T)
    Xf = X.reshape(len(X), -1).astype(np.float64)
    Yf = Y.reshape(len(Y), -1).astype(np.float64)

    if sigma is None:
        # Median heuristic: sigma = median of pairwise distances
        # Use a subsample for efficiency
        rng = np.random.default_rng(42)
        n_sub = min(200, len(Xf), len(Yf))
        Xi = Xf[rng.choice(len(Xf), n_sub, replace=False)]
        Yi = Yf[rng.choice(len(Yf), n_sub, replace=False)]
        XY = np.vstack([Xi, Yi])
        dists = []
        for i in range(0, len(XY), 50):
            diff = XY[i:i+50, None, :] - XY[None, :, :]
            dists.append((diff**2).sum(axis=2).ravel())
        all_dists = np.concatenate(dists)
        sigma = float(np.sqrt(np.median(all_dists[all_dists > 0]) / 2.0))
        sigma = max(sigma, 1e-6)

    # Compute kernel values using batched computation
    def batch_kernel(A, B, sig):
        """Compute mean k(a,b) for a in A, b in B."""
        n_a, n_b = len(A), len(B)
        total = 0.0
        batch = 100
        for i in range(0, n_a, batch):
            Ab = A[i:i+batch]  # (batch, d)
            diff = Ab[:, None, :] - B[None, :, :]  # (batch, n_b, d)
            sq   = (diff**2).sum(axis=2)  # (batch, n_b)
            total += np.exp(-sq / (2*sig**2)).sum()
        return total / (n_a * n_b)

    kxx = batch_kernel(Xf, Xf, sigma)
    kyy = batch_kernel(Yf, Yf, sigma)
    kxy = batch_kernel(Xf, Yf, sigma)
    mmd2 = kxx - 2*kxy + kyy
    return float(mmd2), sigma


def compute_mean_mmd(X_test, X_train_all, n_samples=300, seed=42):
    """
    Compute mean MMD between test subject and all training subjects.
    Uses subsampling for efficiency.
    """
    rng = np.random.default_rng(seed)
    n_te = min(n_samples, len(X_test))
    idx_te = rng.choice(len(X_test), n_te, replace=False)
    X_te_sub = X_test[idx_te]

    n_tr = min(n_samples, len(X_train_all))
    idx_tr = rng.choice(len(X_train_all), n_tr, replace=False)
    X_tr_sub = X_train_all[idx_tr]

    mmd2, sigma = rbf_kernel(X_te_sub, X_tr_sub)
    return float(np.sqrt(max(mmd2, 0.0))), sigma


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',    default='subjects')
    ap.add_argument('--out_dir', default='outputs/plots')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = Path(args.base)
    sds  = sorted([p for p in base.iterdir()
                   if p.is_dir() and p.name.startswith('s')],
                  key=lambda x: int(x.name.replace('s','')))

    print(f"Loading all {len(sds)} subjects...")
    all_X, all_names = {}, []
    for sd in sds:
        emg, lbl, rep = load_subject(sd)
        X, y = make_windows(emg, lbl, rep)
        all_X[sd.name] = X
        all_names.append(sd.name)
        print(f"  {sd.name}: {len(X)} windows")

    # Your confirmed LOSO results (MLP and Latent CNN)
    mlp_bacc = {
        's1':0.07279,'s2':0.06103,'s3':0.06176,'s4':0.05515,'s5':0.05000,
        's6':0.06176,'s7':0.06618,'s8':0.05956,'s9':0.05588,'s10':0.05000,'s11':0.05809
    }
    lat_bacc = {
        's1':0.11838,'s2':0.06103,'s3':0.05735,'s4':0.07059,'s5':0.07132,
        's6':0.06324,'s7':0.05662,'s8':0.06324,'s9':0.08971,'s10':0.06838,'s11':0.09926
    }

    # Compute MMD for each test subject vs pooled training subjects
    print("\nComputing MMD for each LOSO fold...")
    mmd_values = {}
    for test_name in all_names:
        print(f"  Computing MMD for test={test_name}...", flush=True)
        X_test = all_X[test_name]
        X_train_list = [all_X[n] for n in all_names if n != test_name]
        X_train_all  = np.concatenate(X_train_list)

        mmd, sigma = compute_mean_mmd(X_test, X_train_all, n_samples=300)
        mmd_values[test_name] = mmd
        print(f"    {test_name}: MMD={mmd:.4f}  sigma={sigma:.1f}")

    # Print summary
    print("\n=== MMD Summary ===")
    for s in all_names:
        print(f"  {s}: MMD={mmd_values[s]:.4f}  "
              f"MLP_bacc={mlp_bacc[s]:.4f}  Latent_bacc={lat_bacc[s]:.4f}")

    # ── Plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    C_RED    = '#e74c3c'
    C_PURPLE = '#9b59b6'
    C_NAVY   = '#1F4E79'

    subjects = all_names
    mmd_x   = [mmd_values[s] for s in subjects]
    mlp_y   = [mlp_bacc[s]   for s in subjects]
    lat_y   = [lat_bacc[s]   for s in subjects]

    # Left: MMD vs accuracy scatter
    ax = axes[0]
    ax.scatter(mmd_x, mlp_y, c=C_RED,    s=70, zorder=3, alpha=0.85,
               label='MLP SoftMax', marker='o')
    ax.scatter(mmd_x, lat_y, c=C_PURPLE, s=70, zorder=3, alpha=0.85,
               label='Latent CNN',  marker='D')
    for i, s in enumerate(subjects):
        y_pos = max(mlp_y[i], lat_y[i]) + 0.002
        ax.annotate(s.upper(), (mmd_x[i], y_pos),
                    ha='center', fontsize=8, color='#444')

    # Trend lines
    for y_vals, col in [(mlp_y, C_RED), (lat_y, C_PURPLE)]:
        z = np.polyfit(mmd_x, y_vals, 1)
        px = np.linspace(min(mmd_x)*0.95, max(mmd_x)*1.05, 50)
        ax.plot(px, np.poly1d(z)(px), color=col,
                linestyle='--', alpha=0.5, lw=1.5)

    ax.set_xlabel('Maximum Mean Discrepancy (MMD)\nbetween test subject and training pool',
                  fontsize=10)
    ax.set_ylabel('LOSO Balanced Accuracy', fontsize=10)
    ax.set_title('Domain shift (MMD) vs LOSO accuracy\nHigher MMD = harder generalization',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, zorder=0); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Right: MMD per subject bar chart
    ax2 = axes[1]
    x = np.arange(len(subjects))
    bars = ax2.bar(x, mmd_x, color=C_NAVY, edgecolor='white',
                   linewidth=1, zorder=3, width=0.6)
    # Highlight highest MMD (hardest subject)
    max_idx = int(np.argmax(mmd_x))
    bars[max_idx].set_color('#e74c3c')
    for bar, val in zip(bars, mmd_x):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.upper() for s in subjects], fontsize=9)
    ax2.set_ylabel('MMD (test vs training pool)', fontsize=10)
    ax2.set_title('Per-subject domain shift (MMD)\nRed = highest shift, hardest subject',
                  fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, zorder=0); ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.suptitle('Maximum Mean Discrepancy as a proper domain shift metric\n'
                 'NinaPro DB3 — inter-subject distribution distance vs LOSO accuracy',
                 fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, 'fig_mmd_domain_shift.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")
    print("Replace fig3_domain_shift.png with this figure in your paper.")
    print("This uses proper MMD metric as requested by your professor.")

if __name__ == '__main__':
    main()
