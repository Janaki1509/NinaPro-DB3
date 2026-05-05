#!/usr/bin/env python3
"""
run_protocol_a_vs_b.py  (corrected version)
============================================
Protocol A: ALL windows, random 80/20 split WITHIN each subject.
            No per-class cap — uses every window available.
            This matches how published papers report 70-90% accuracy.

Protocol B: LOSO with balanced subsampling (same as your benchmark).
            Uses confirmed results from your existing CSVs.

The key insight: Protocol A is easy because train and test windows
come from the SAME subject. Protocol B is hard because test subject
is completely unseen. Same model, same preprocessing, different split.
"""

import argparse, os
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch, torch.nn as nn
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

def channel_norm(X):
    m = X.mean(axis=(0,2),keepdims=True).astype(np.float32)
    s = X.std(axis=(0,2),keepdims=True)
    return m, np.where(s<1e-6,1.0,s).astype(np.float32)

def remap(y_tr, y_te):
    cls = np.sort(np.unique(np.concatenate([y_tr, y_te])))
    c2i = {int(c):i for i,c in enumerate(cls)}
    return (np.array([c2i[int(c)] for c in y_tr], np.int64),
            np.array([c2i[int(c)] for c in y_te], np.int64),
            len(cls))


# ── MLP ───────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,K))
    def forward(self, x): return self.net(x)

def train_eval_uncapped(X_tr, y_tr, X_te, y_te, epochs=20):
    """Protocol A: NO subsampling cap — use ALL windows."""
    common = np.intersect1d(np.unique(y_tr), np.unique(y_te))
    X_tr = X_tr[np.isin(y_tr,common)]; y_tr = y_tr[np.isin(y_tr,common)]
    X_te = X_te[np.isin(y_te,common)]; y_te = y_te[np.isin(y_te,common)]
    if len(common) == 0: return None, None, None
    m, s = channel_norm(X_tr)
    X_tr = (X_tr-m)/s; X_te = (X_te-m)/s
    N, C, T = X_tr.shape; K = len(common)
    y_tr_i, y_te_i, _ = remap(y_tr, y_te)
    Xf_tr = torch.from_numpy(X_tr.reshape(N, C*T))
    Xf_te = torch.from_numpy(X_te.reshape(len(X_te), C*T))
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
    return balanced_accuracy_score(y_te_i, preds), accuracy_score(y_te_i, preds), K


# ── Protocol A ────────────────────────────────────────────────────────────

def run_protocol_a(subject_dirs, epochs=20):
    results = []
    for sd in subject_dirs:
        print(f'  Protocol A: {sd.name}...', flush=True)
        emg, lbl, rep = load_subject(sd)
        X, y = make_windows(emg, lbl, rep)
        if len(X) == 0: continue
        # Random 80/20 split — NO cap on windows
        rng = np.random.default_rng(42)
        idx = np.arange(len(X)); rng.shuffle(idx)
        split = int(0.8*len(idx))
        tr, te = idx[:split], idx[split:]
        bacc, acc, K = train_eval_uncapped(X[tr], y[tr], X[te], y[te], epochs)
        if bacc is None: continue
        print(f'    {sd.name}: bacc={bacc:.4f}  acc={acc:.4f}  '
              f'train_windows={len(tr)}  test_windows={len(te)}  K={K}')
        results.append({'subject':sd.name,'bacc':bacc,'acc':acc,'K':K,
                        'n_train':len(tr),'n_test':len(te)})
    return results


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_main_comparison(a_mean, b_mean, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: side-by-side bars
    ax = axes[0]
    labels = ['Protocol A\n(Within-subject\npooled 80/20)', 'Protocol B\n(LOSO\ncross-subject)']
    vals   = [a_mean, b_mean]
    colors = ['#2980b9', '#e74c3c']
    bars = ax.bar(labels, vals, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.45, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.012,
                f'{val:.3f}\n({val:.1%})',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.axhline(0.15, color='#888', linestyle='--', lw=1.2,
               label='LOSO target (0.15)', alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Mean Balanced Accuracy (all 11 subjects)', fontsize=11)
    ax.set_title('Same model, same preprocessing\nDifferent evaluation split',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.35, zorder=0); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Right: ratio / gap explanation
    ax2 = axes[1]
    ratio = a_mean / b_mean
    gap   = a_mean - b_mean
    ax2.barh(['Protocol A\n(same subject)','Protocol B\n(unseen subject)'],
             [a_mean, b_mean], color=['#2980b9','#e74c3c'],
             edgecolor='white', height=0.35, zorder=3)
    for i, val in enumerate([a_mean, b_mean]):
        ax2.text(val+0.01, i, f'{val:.3f}  ({val:.1%})',
                 va='center', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(a_mean*1.35, 0.5))
    ax2.set_xlabel('Mean Balanced Accuracy', fontsize=11)
    ax2.set_title(f'Cross-subject is {ratio:.1f}× harder\nGap = {gap:.3f} ({gap:.1%})',
                  fontsize=12, fontweight='bold')
    ax2.axvline(0.15, color='#888', linestyle='--', lw=1.2, alpha=0.8)
    ax2.grid(axis='x', alpha=0.35, zorder=0); ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    fig.suptitle(
        'Protocol sensitivity: within-subject vs cross-subject evaluation\n'
        'NinaPro DB3, MLP SoftMax, 11 subjects, 17 gesture classes',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def plot_per_subject(a_results, b_loso_dict, out_path):
    a_dict = {r['subject']: r['bacc'] for r in a_results}
    subjects = sorted(a_dict.keys(), key=lambda x: int(x.replace('s','')))
    x = np.arange(len(subjects)); w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x-w/2, [a_dict.get(s,0)      for s in subjects], w,
           label='Protocol A (within-subject, uncapped)',
           color='#2980b9', edgecolor='white', zorder=3)
    ax.bar(x+w/2, [b_loso_dict.get(s,0) for s in subjects], w,
           label='Protocol B (LOSO, balanced subsample)',
           color='#e74c3c', edgecolor='white', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in subjects], fontsize=10)
    ax.set_ylabel('Balanced Accuracy', fontsize=11)
    ax.set_title('Per-subject comparison: Protocol A vs Protocol B\n'
                 'Same MLP model — only the evaluation split differs',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis='y', alpha=0.35, zorder=0); ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────

# Your confirmed LOSO MLP results (Protocol B) — no need to rerun
CONFIRMED_LOSO = {
    's1':0.07279,'s2':0.06103,'s3':0.06176,'s4':0.05515,'s5':0.05000,
    's6':0.06176,'s7':0.06618,'s8':0.05956,'s9':0.05588,'s10':0.05000,'s11':0.05809
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',    default='subjects')
    ap.add_argument('--out_dir', default='outputs/plots')
    ap.add_argument('--epochs',  type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = Path(args.base)
    sds  = sorted([p for p in base.iterdir()
                   if p.is_dir() and p.name.startswith('s')],
                  key=lambda x: int(x.name.replace('s','')))

    print(f'Running Protocol A on {len(sds)} subjects (NO window cap)...')
    a_results = run_protocol_a(sds, args.epochs)

    a_mean = np.mean([r['bacc'] for r in a_results])
    b_mean = np.mean(list(CONFIRMED_LOSO.values()))

    print(f'\n=== FINAL RESULTS ===')
    print(f'Protocol A (within-subject, uncapped): {a_mean:.4f} ({a_mean:.1%})')
    print(f'Protocol B (LOSO, confirmed):          {b_mean:.4f} ({b_mean:.1%})')
    print(f'Ratio: {a_mean/b_mean:.1f}x harder cross-subject')
    print(f'Gap:   {a_mean-b_mean:.4f}')

    plot_main_comparison(a_mean, b_mean,
        os.path.join(args.out_dir, 'protocol_a_vs_b.png'))
    plot_per_subject(a_results, CONFIRMED_LOSO,
        os.path.join(args.out_dir, 'protocol_a_vs_b_per_subject.png'))

    print('\nDone. These are your two key Protocol A vs B figures.')

if __name__ == '__main__':
    main()
