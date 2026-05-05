#!/usr/bin/env python3
import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    rec = []
    for k in range(K):
        m = (y_true == k)
        if m.any():
            rec.append(float((y_pred[m] == k).mean()))
    return float(np.mean(rec)) if rec else 0.0

def stratified_split(y: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    by = {}
    for i, c in enumerate(y):
        by.setdefault(int(c), []).append(i)
    tr, va = [], []
    for _, idxs in by.items():
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_frac))
        va.append(idxs[:n_val])
        tr.append(idxs[n_val:])
    tr = np.concatenate(tr) if tr else np.array([], dtype=np.int64)
    va = np.concatenate(va) if va else np.array([], dtype=np.int64)
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va

class EMGWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)  # (N,C,T)
        self.y = y.astype(np.int64)

    def __len__(self): return self.y.shape[0]

    def __getitem__(self, i):
        x = self.X[i]                  # (C,T)
        x = np.transpose(x, (1, 0))    # (T,C)
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=12, hidden=64, layers=1, dropout=0.2, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden,
            num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x):
        _, (h, _) = self.lstm(x)   # h: (layers,B,H)
        return self.head(h[-1])    # (B,K)

@torch.no_grad()
def evaluate(model, loader, criterion, device, K):
    model.eval()
    losses, ys, ps = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        pred = torch.argmax(logits, dim=1)
        ys.append(yb.cpu().numpy()); ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    acc = float((y_true == y_pred).mean())
    bacc = balanced_accuracy(y_true, y_pred, K)
    return float(np.mean(losses)) if losses else 0.0, acc, bacc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"data\folds\within_s1_rep.npz")
    ap.add_argument("--outdir", default=r"runs\lstm_v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    d = np.load(args.data)
    X_train = d["X_train"].astype(np.float32)  # (N,C,T)
    y_train = d["y_train"].astype(np.int32)
    X_test  = d["X_test"].astype(np.float32)
    y_test  = d["y_test"].astype(np.int32)

    # label map using TRAIN labels
    labels = np.unique(y_train)
    l2i = {int(lbl): i for i, lbl in enumerate(labels)}
    ytr = np.array([l2i[int(v)] for v in y_train], dtype=np.int64)
    yte = np.array([l2i[int(v)] for v in y_test], dtype=np.int64)
    K = len(labels)

    # per-channel standardization using TRAIN only (global)
    mu = X_train.mean(axis=(0,2), keepdims=True)  # (1,C,1)
    sd = X_train.std(axis=(0,2), keepdims=True) + 1e-8
    X_train = (X_train - mu) / sd
    X_test  = (X_test  - mu) / sd

    tr_idx, va_idx = stratified_split(ytr, args.val_frac, args.seed)
    Xtr, ytr2 = X_train[tr_idx], ytr[tr_idx]
    Xva, yva  = X_train[va_idx], ytr[va_idx]

    train_loader = DataLoader(EMGWindowDataset(Xtr, ytr2), batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(EMGWindowDataset(Xva, yva),  batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(EMGWindowDataset(X_test, yte),batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(12, args.hidden, args.layers, args.dropout, K).to(device)

    # class weights
    counts = np.bincount(ytr2, minlength=K).astype(np.float32)
    w = (counts.sum() / (counts + 1e-6)); w = w / w.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.outdir, exist_ok=True)
    best_path = os.path.join(args.outdir, "best.pt")
    meta_path = os.path.join(args.outdir, "meta.json")

    best_val_bacc = -1.0
    print(f"Device: {device}")
    print(f"Train {len(Xtr)} | Val {len(Xva)} | Test {len(X_test)} | K={K}")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())

        va_loss, va_acc, va_bacc = evaluate(model, val_loader, criterion, device, K)
        print(f"ep {ep:03d} | va acc {va_acc:.3f} | va bacc {va_bacc:.3f}")

        if va_bacc > best_val_bacc:
            best_val_bacc = va_bacc
            torch.save({"state_dict": model.state_dict(), "labels": labels, "mu": mu, "sd": sd}, best_path)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_acc, te_bacc = evaluate(model, test_loader, criterion, device, K)
    print("\nBEST -> TEST:")
    print(f"test acc {te_acc:.3f} | test bacc {te_bacc:.3f}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data": args.data, "K": K, "best_val_bacc": best_val_bacc,
                   "test_acc": te_acc, "test_bacc": te_bacc}, f, indent=2)

if __name__ == "__main__":
    main()
