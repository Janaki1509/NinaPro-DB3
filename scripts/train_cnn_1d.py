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

def per_window_norm(X):
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mu) / sd

class EMGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, gain_aug: float = 0.0):
        self.X = X.astype(np.float32)  # (N,C,T)
        self.y = y.astype(np.int64)
        self.g = float(gain_aug)

    def __len__(self): return self.y.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        if self.g > 0:
            g = np.random.uniform(1.0 - self.g, 1.0 + self.g)
            x = x * np.float32(g)
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

class SmallCNN(nn.Module):
    def __init__(self, C=12, K=38, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),  # -> (B,256,1)
            nn.Flatten(),             # -> (B,256)
            nn.Dropout(dropout),
            nn.Linear(256, K)
        )

    def forward(self, x):
        return self.net(x)

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
    ap.add_argument("--data", default=r"data\folds\loso_s1_bigger.npz")
    ap.add_argument("--outdir", default=r"runs\cnn1d_run")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--restrict_to_test_labels", type=int, default=1)
    ap.add_argument("--use_per_window_norm", type=int, default=1)
    ap.add_argument("--gain_aug", type=float, default=0.3)
    args = ap.parse_args()

    set_seed(args.seed)

    d = np.load(args.data)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int32)
    X_test  = d["X_test"].astype(np.float32)
    y_test  = d["y_test"].astype(np.int32)

    if args.restrict_to_test_labels == 1:
        test_labels = np.unique(y_test)
        m = np.isin(y_train, test_labels)
        X_train, y_train = X_train[m], y_train[m]
        print("Restricted train classes:", len(np.unique(y_train)), "test classes:", len(np.unique(y_test)))

    labels = np.unique(y_train)
    l2i = {int(lbl): i for i, lbl in enumerate(labels)}
    ytr = np.array([l2i[int(v)] for v in y_train], dtype=np.int64)
    yte = np.array([l2i[int(v)] for v in y_test], dtype=np.int64)
    K = len(labels)

    if args.use_per_window_norm == 1:
        X_train = per_window_norm(X_train)
        X_test  = per_window_norm(X_test)

    tr_idx, va_idx = stratified_split(ytr, args.val_frac, args.seed)
    Xtr, ytr2 = X_train[tr_idx], ytr[tr_idx]
    Xva, yva  = X_train[va_idx], ytr[va_idx]

    train_loader = DataLoader(EMGDataset(Xtr, ytr2, gain_aug=args.gain_aug),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(EMGDataset(Xva, yva, gain_aug=0.0),
                              batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(EMGDataset(X_test, yte, gain_aug=0.0),
                              batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(C=12, K=K, dropout=args.dropout).to(device)

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
            optim.step()
            losses.append(loss.item())

        va_loss, va_acc, va_bacc = evaluate(model, val_loader, criterion, device, K)
        print(f"ep {ep:03d} | va acc {va_acc:.3f} | va bacc {va_bacc:.3f}")

        if va_bacc > best_val_bacc:
            best_val_bacc = va_bacc
            torch.save({"state_dict": model.state_dict(), "labels": labels, "args": vars(args)}, best_path)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_acc, te_bacc = evaluate(model, test_loader, criterion, device, K)
    print("\nBEST -> TEST:")
    print(f"test acc {te_acc:.3f} | test bacc {te_bacc:.3f}")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"data": args.data, "K": K, "best_val_bacc": best_val_bacc,
                   "test_acc": te_acc, "test_bacc": te_bacc}, f, indent=2)
    print("Saved:", best_path)
    print("Saved:", meta_path)

if __name__ == "__main__":
    main()
