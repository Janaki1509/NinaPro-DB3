#!/usr/bin/env python3
"""
run_trust_calibration.py
=========================
Steps 5, 6, 7 in one script — runs on your LAPTOP, no HPC needed.

Produces:
  reliability_before.png       — Step 7: overconfidence before calibration
  reliability_after.png        — Step 7: after temperature scaling
  accuracy_vs_confidence.png   — Step 5: trust score validity check
  risk_coverage_curve.png      — Step 6: main clinical innovation figure
  accuracy_vs_coverage.png     — Step 6: usability vs accuracy trade-off
  calibration_summary.csv      — ECE, Brier, recommended tau

Usage (Windows PowerShell from C:\\ninapro_db3):
  python scripts\\run_trust_calibration.py `
      --base subjects `
      --test s1 `
      --model mlp `
      --out_dir outputs\\calibration

Run for both models:
  --model mlp
  --model latent
"""

import argparse, csv, os
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Data loading (identical to your existing scripts) ─────────────────────

def _maybe_array(x):
    try:
        arr = np.asarray(x)
        if arr.size > 0: return arr
    except: pass
    return None

def _search_for_key(obj, keys, depth=0, max_depth=4):
    if depth > max_depth: return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                a = _maybe_array(v)
                if a is not None: return a
        for _, v in obj.items():
            f = _search_for_key(v, keys, depth+1, max_depth)
            if f is not None: return f
    if hasattr(obj, '__dict__'):
        for k, v in obj.__dict__.items():
            if k in keys:
                a = _maybe_array(v)
                if a is not None: return a
        for _, v in obj.__dict__.items():
            f = _search_for_key(v, keys, depth+1, max_depth)
            if f is not None: return f
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for item in obj.flat:
            f = _search_for_key(item, keys, depth+1, max_depth)
            if f is not None: return f
    return None

def load_subject(sd):
    mats = sorted(Path(sd).rglob('*.mat.mat')) or sorted(Path(sd).rglob('*.mat'))
    if not mats: raise FileNotFoundError(sd)
    data = sio.loadmat(str(mats[0]), squeeze_me=True, struct_as_record=False)
    emg  = np.asarray(_search_for_key(data, {'emg'}))
    stim = np.asarray(_search_for_key(data, {'restimulus'})).reshape(-1)
    rep  = np.asarray(_search_for_key(data, {'rerepetition'})).reshape(-1)
    return emg.astype(np.float32), stim.astype(np.int64), rep.astype(np.int64)

def make_windows(emg, labels, reps, win_len, step):
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

def subsample(X, y, n, rng):
    idx = []
    for c in np.unique(y):
        i = np.flatnonzero(y==c)
        idx.append(rng.choice(i, min(len(i),n), replace=False))
    idx = np.concatenate(idx); rng.shuffle(idx)
    return X[idx], y[idx]

def common_classes(Xtr,ytr,Xte,yte):
    c = np.intersect1d(np.unique(ytr), np.unique(yte))
    return Xtr[np.isin(ytr,c)],ytr[np.isin(ytr,c)],Xte[np.isin(yte,c)],yte[np.isin(yte,c)],c

def channel_norm(X):
    m = X.mean(axis=(0,2),keepdims=True).astype(np.float32)
    s = X.std(axis=(0,2),keepdims=True)
    return m, np.where(s<1e-6,1.0,s).astype(np.float32)


# ── Models ────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, d, h, K, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(p),nn.Linear(h,K))
    def forward(self, x): return self.net(x), None

class ConvNet(nn.Module):
    def __init__(self, C, T, ld=128, K=52, p=0.3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(C,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(128,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2))
        self.bot = nn.Sequential(nn.Flatten(),nn.Linear(128*(T//8),ld),nn.ReLU(),nn.Dropout(p))
        self.clf = nn.Linear(ld,K)
    def forward(self, x):
        z = self.bot(self.enc(x)); return self.clf(z), z


# ── Calibration helpers ───────────────────────────────────────────────────

def softmax(logits):
    e = np.exp(logits - logits.max(1,keepdims=True))
    return e / e.sum(1,keepdims=True)

def fit_temperature(logits_val, labels_val):
    T = nn.Parameter(torch.ones(1)*1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=300)
    lv = torch.from_numpy(logits_val).float()
    lbl = torch.from_numpy(labels_val).long()
    crit = nn.CrossEntropyLoss()
    def step():
        opt.zero_grad()
        loss = crit(lv/T, lbl)
        loss.backward()
        return loss
    opt.step(step)
    return max(0.1, float(T.item()))

def ece(probs, labels, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1)
    correct = (pred==labels).astype(float)
    bins = np.linspace(0,1,n_bins+1)
    ece_val = 0.0
    for i in range(n_bins):
        m = (conf>=bins[i])&(conf<bins[i+1])
        if m.sum()==0: continue
        ece_val += m.sum()*abs(correct[m].mean()-conf[m].mean())
    return float(ece_val/len(labels))

def brier(probs, labels):
    oh = np.zeros_like(probs)
    oh[np.arange(len(labels)),labels] = 1.0
    return float(np.mean(np.sum((probs-oh)**2,1)))


# ── Plot helpers ──────────────────────────────────────────────────────────

def plot_reliability_diagram(probs, labels, title, out_path, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1)
    correct = (pred==labels).astype(float)
    bins = np.linspace(0,1,n_bins+1)
    centers, accs, gaps = [], [], []
    for i in range(n_bins):
        m = (conf>=bins[i])&(conf<bins[i+1])
        c = (bins[i]+bins[i+1])/2; centers.append(c)
        if m.sum()==0: accs.append(0); gaps.append(0)
        else:
            a = correct[m].mean(); accs.append(a)
            gaps.append(max(0, conf[m].mean()-a))

    fig, ax = plt.subplots(figsize=(5.5,5.5))
    ax.bar(centers, accs,  width=0.09, color='#5b7fce', alpha=0.85,
           edgecolor='white', zorder=3, label='Accuracy')
    ax.bar(centers, gaps,  width=0.09, color='#e06050', alpha=0.55,
           bottom=accs, edgecolor='white', zorder=3, label='Overconfidence gap')
    ax.plot([0,1],[0,1],'--',color='#555',lw=1.5,label='Perfect calibration')
    e = ece(probs,labels); b = brier(probs,labels)
    ax.set_title(f'{title}\nECE = {e:.4f}   Brier = {b:.4f}',
                 fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Confidence (max softmax probability)', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    return e, b

def plot_accuracy_vs_confidence(probs, labels, model_name, out_path, n_bins=10):
    """Step 5: when trust score is high, is accuracy actually higher?"""
    conf = probs.max(1); pred = probs.argmax(1)
    correct = (pred==labels).astype(float)
    bins = np.linspace(0,1,n_bins+1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        m = (conf>=bins[i])&(conf<bins[i+1])
        centers.append((bins[i]+bins[i+1])/2)
        accs.append(correct[m].mean() if m.sum()>0 else np.nan)
        counts.append(m.sum())

    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()
    valid = [i for i,a in enumerate(accs) if not np.isnan(a)]
    ax1.plot([centers[i] for i in valid], [accs[i] for i in valid],
             'o-', color='#5b7fce', lw=2, markersize=6, label='Accuracy', zorder=4)
    ax2.bar(centers, counts, width=0.08, color='#9b59b6', alpha=0.3,
            label='Sample count', zorder=2)
    ax1.plot([0,1],[0,1],'--',color='#aaa',lw=1,label='Perfect calibration')
    ax1.set_xlabel('Trust score (max softmax probability)', fontsize=11)
    ax1.set_ylabel('Accuracy in bin', fontsize=11, color='#5b7fce')
    ax2.set_ylabel('Number of predictions', fontsize=11, color='#9b59b6')
    ax1.set_xlim(0,1); ax1.set_ylim(0,1)
    ax1.set_title(f'{model_name} — accuracy vs trust score\n'
                  'Higher confidence → higher accuracy (validates trust score)',
                  fontsize=11, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, fontsize=9, loc='upper left')
    ax1.grid(alpha=0.3); ax1.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()

def plot_risk_coverage(probs, labels, model_name, out_path_rc, out_path_ac):
    """Step 6: abstention / pause-when-unsure analysis."""
    conf = probs.max(1); pred = probs.argmax(1)
    correct = (pred==labels).astype(float)
    taus = np.linspace(0.05, 0.99, 100)
    coverages, risks, accs_covered = [], [], []
    for tau in taus:
        mask = conf >= tau
        cov  = mask.mean()
        if mask.sum() == 0:
            coverages.append(cov); risks.append(np.nan); accs_covered.append(np.nan)
        else:
            coverages.append(cov)
            risks.append(1 - correct[mask].mean())
            accs_covered.append(correct[mask].mean())

    # Find recommended tau: risk < 0.5 with highest coverage
    best_tau, best_cov = None, 0.0
    for tau, cov, risk in zip(taus, coverages, risks):
        if risk is not None and not np.isnan(risk) and risk < 0.5 and cov > best_cov:
            best_tau, best_cov = tau, cov

    # Risk-coverage curve
    fig, ax = plt.subplots(figsize=(7,5))
    valid = [(c,r) for c,r in zip(coverages,risks) if not np.isnan(r)]
    cvs, rks = zip(*valid) if valid else ([],[])
    ax.plot(cvs, rks, 'o-', color='#e06050', lw=2, markersize=3, alpha=0.8)
    if best_tau is not None:
        bt_idx = np.argmin(np.abs(taus - best_tau))
        ax.scatter([coverages[bt_idx]], [risks[bt_idx]],
                   s=120, color='#27500A', zorder=5,
                   label=f'Recommended τ={best_tau:.2f}\n(coverage={best_cov:.2f})')
        ax.legend(fontsize=9)
    ax.axhline(0.5, color='#aaa', lw=1, linestyle='--', label='50% error rate')
    ax.set_xlabel('Coverage (fraction of predictions acted on)', fontsize=11)
    ax.set_ylabel('Risk (error rate when acting)', fontsize=11)
    ax.set_title(f'{model_name} — risk–coverage curve\n'
                 'Lower-right = safer: less risk at same coverage',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path_rc, dpi=180, bbox_inches='tight')
    plt.close()

    # Accuracy vs coverage
    fig, ax = plt.subplots(figsize=(7,5))
    valid2 = [(c,a) for c,a in zip(coverages,accs_covered) if not np.isnan(a)]
    cvs2, acs2 = zip(*valid2) if valid2 else ([],[])
    ax.plot(cvs2, acs2, 'o-', color='#5b7fce', lw=2, markersize=3, alpha=0.8)
    if best_tau is not None:
        bt_idx = np.argmin(np.abs(taus - best_tau))
        if not np.isnan(accs_covered[bt_idx]):
            ax.scatter([coverages[bt_idx]], [accs_covered[bt_idx]],
                       s=120, color='#27500A', zorder=5,
                       label=f'Recommended τ={best_tau:.2f}')
            ax.legend(fontsize=9)
    ax.set_xlabel('Coverage (fraction of predictions acted on)', fontsize=11)
    ax.set_ylabel('Accuracy when acting', fontsize=11)
    ax.set_title(f'{model_name} — accuracy vs coverage\n'
                 'Trade-off: act less often, but more accurately',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path_ac, dpi=180, bbox_inches='tight')
    plt.close()
    return best_tau, best_cov


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',    required=True)
    ap.add_argument('--test',    default='s1')
    ap.add_argument('--model',   default='mlp', choices=['mlp','latent'])
    ap.add_argument('--out_dir', default='./outputs/calibration')
    ap.add_argument('--epochs',  type=int,   default=30)
    ap.add_argument('--train_per_class', type=int, default=120)
    ap.add_argument('--test_per_class',  type=int, default=80)
    ap.add_argument('--seed',    type=int,   default=42)
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cpu')
    model_name = 'MLP SoftMax' if args.model=='mlp' else 'Latent CNN'

    base = Path(args.base)
    sds  = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith('s')])
    win_len = 300; step = 150

    Xtr_all,ytr_all,Xte_all,yte_all = [],[],[],[]
    for sd in sds:
        emg,lbl,rep = load_subject(sd)
        X,y = make_windows(emg,lbl,rep,win_len,step)
        (Xte_all if sd.name==args.test else Xtr_all).append(X)
        (yte_all if sd.name==args.test else ytr_all).append(y)

    Xtr=np.concatenate(Xtr_all); ytr=np.concatenate(ytr_all)
    Xte=np.concatenate(Xte_all); yte=np.concatenate(yte_all)
    Xtr,ytr,Xte,yte,_ = common_classes(Xtr,ytr,Xte,yte)
    Xtr,ytr = subsample(Xtr,ytr,args.train_per_class,rng)

    # 80/20 val split from train
    n = len(Xtr)
    vi = rng.choice(n,int(n*0.2),replace=False)
    ti = np.setdiff1d(np.arange(n),vi)
    Xva,yva = Xtr[vi],ytr[vi]; Xtr,ytr = Xtr[ti],ytr[ti]
    Xte,yte = subsample(Xte,yte,args.test_per_class,rng)

    cls = np.sort(np.unique(np.concatenate([ytr,yva,yte])))
    c2i = {int(c):i for i,c in enumerate(cls)}
    ytr_i = np.array([c2i[int(c)] for c in ytr],np.int64)
    yva_i = np.array([c2i[int(c)] for c in yva],np.int64)
    yte_i = np.array([c2i[int(c)] for c in yte],np.int64)
    K = len(cls)

    m,s = channel_norm(Xtr)
    Xtr=(Xtr-m)/s; Xva=(Xva-m)/s; Xte=(Xte-m)/s
    N,C,T = Xtr.shape

    print(f'[{model_name}] Train={N} Val={len(Xva)} Test={len(Xte)} K={K}')

    if args.model=='mlp':
        Xtr_t=torch.from_numpy(Xtr.reshape(N,C*T))
        Xva_t=torch.from_numpy(Xva.reshape(len(Xva),C*T))
        Xte_t=torch.from_numpy(Xte.reshape(len(Xte),C*T))
        model = MLP(C*T,128,K).to(device)
    else:
        Xtr_t=torch.from_numpy(Xtr); Xva_t=torch.from_numpy(Xva); Xte_t=torch.from_numpy(Xte)
        model = ConvNet(C,T,128,K).to(device)

    opt = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr_t,torch.from_numpy(ytr_i)),
                        batch_size=256,shuffle=True)
    print(f'Training {args.epochs} epochs...')
    model.train()
    for ep in range(args.epochs):
        for xb,yb in loader:
            opt.zero_grad()
            out = model(xb); logits = out[0] if isinstance(out,tuple) else out
            crit(logits,yb).backward(); opt.step()
        if (ep+1)%10==0: print(f'  epoch {ep+1}',flush=True)

    model.eval()
    with torch.no_grad():
        lv  = model(Xva_t); lv  = (lv[0]  if isinstance(lv,tuple)  else lv).numpy()
        lte = model(Xte_t); lte = (lte[0] if isinstance(lte,tuple) else lte).numpy()

    # ── Step 7: calibration ────────────────────────────────────────────────
    probs_raw = softmax(lte)
    print('\nStep 7 — Calibration')
    ece_b = ece(probs_raw,yte_i); bs_b = brier(probs_raw,yte_i)
    plot_reliability_diagram(probs_raw, yte_i,
        f'{model_name} — before calibration',
        os.path.join(args.out_dir,f'reliability_before_{args.model}.png'))
    print(f'  Before: ECE={ece_b:.4f}  Brier={bs_b:.4f}')

    T_opt = fit_temperature(lv, yva_i)
    probs_cal = softmax(lte/T_opt)
    ece_a = ece(probs_cal,yte_i); bs_a = brier(probs_cal,yte_i)
    plot_reliability_diagram(probs_cal, yte_i,
        f'{model_name} — after temperature scaling (T={T_opt:.2f})',
        os.path.join(args.out_dir,f'reliability_after_{args.model}.png'))
    print(f'  After:  ECE={ece_a:.4f}  Brier={bs_a:.4f}  T={T_opt:.4f}')

    # ── Step 5: accuracy vs confidence ────────────────────────────────────
    print('\nStep 5 — Accuracy vs trust score')
    plot_accuracy_vs_confidence(probs_cal, yte_i, model_name,
        os.path.join(args.out_dir,f'accuracy_vs_confidence_{args.model}.png'))
    print('  Saved accuracy_vs_confidence plot')

    # ── Step 6: risk-coverage (abstention) ────────────────────────────────
    print('\nStep 6 — Risk-coverage / abstention')
    best_tau, best_cov = plot_risk_coverage(
        probs_cal, yte_i, model_name,
        os.path.join(args.out_dir,f'risk_coverage_{args.model}.png'),
        os.path.join(args.out_dir,f'accuracy_vs_coverage_{args.model}.png'))
    print(f'  Recommended tau={best_tau}  coverage={best_cov:.2f}')

    # ── Summary CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir,'calibration_summary.csv')
    write_hdr = not os.path.exists(csv_path)
    with open(csv_path,'a',newline='') as f:
        w = csv.writer(f)
        if write_hdr:
            w.writerow(['model','test_subject','temperature',
                        'ece_before','brier_before',
                        'ece_after','brier_after','recommended_tau'])
        w.writerow([args.model, args.test, f'{T_opt:.4f}',
                    f'{ece_b:.4f}',f'{bs_b:.4f}',
                    f'{ece_a:.4f}',f'{bs_a:.4f}',
                    f'{best_tau:.2f}' if best_tau else 'N/A'])

    print(f'\nAll figures saved to: {args.out_dir}')
    print(f'Summary: ECE {ece_b:.4f} -> {ece_a:.4f} | '
          f'Brier {bs_b:.4f} -> {bs_a:.4f} | tau={best_tau}')

if __name__=='__main__':
    main()
