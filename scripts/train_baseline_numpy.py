#!/usr/bin/env python3
import argparse
import numpy as np

# ---------- feature extraction (fast + standard EMG time-domain features) ----------
def emg_features(X, zc_thr=1e-3, ssc_thr=1e-3):
    """
    X: (N, C, T) float32
    returns: (N, C*5) features [MAV, RMS, WL, ZC, SSC] per channel
    """
    N, C, T = X.shape
    eps = 1e-8

    mav = np.mean(np.abs(X), axis=2)                               # (N,C)
    rms = np.sqrt(np.mean(X * X, axis=2) + eps)                    # (N,C)
    wl  = np.sum(np.abs(np.diff(X, axis=2)), axis=2)               # (N,C)

    # Zero crossings (ZC)
    x1 = X[:, :, :-1]
    x2 = X[:, :,  1:]
    sign_change = (x1 * x2) < 0
    amp_ok = np.abs(x1 - x2) > zc_thr
    zc = np.sum(sign_change & amp_ok, axis=2).astype(np.float32)   # (N,C)

    # Slope sign changes (SSC)
    d1 = X[:, :, 1:-1] - X[:, :, :-2]
    d2 = X[:, :, 2:  ] - X[:, :, 1:-1]
    ssc = np.sum(((d1 * d2) < 0) & ((np.abs(d1) > ssc_thr) | (np.abs(d2) > ssc_thr)), axis=2).astype(np.float32)

    feats = np.concatenate([mav, rms, wl, zc, ssc], axis=1)         # (N, 5C)
    return feats.astype(np.float32)

def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

# ---------- softmax regression (multiclass logistic regression) ----------
def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)

def one_hot(y, K):
    oh = np.zeros((y.size, K), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def balanced_accuracy(y_true, y_pred, K):
    # mean recall over classes present in y_true
    recalls = []
    for k in range(K):
        mask = (y_true == k)
        if mask.any():
            recalls.append(np.mean(y_pred[mask] == k))
    return float(np.mean(recalls)) if recalls else 0.0

def train_softmax_regression(X, y, K, lr=0.05, l2=1e-3, epochs=50, batch_size=256, seed=42):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    W = 0.01 * rng.standard_normal((D, K)).astype(np.float32)
    b = np.zeros((1, K), dtype=np.float32)

    y_oh = one_hot(y, K)

    for ep in range(1, epochs + 1):
        idx = rng.permutation(N)
        Xs = X[idx]
        ys = y[idx]
        yohs = y_oh[idx]

        for i in range(0, N, batch_size):
            xb = Xs[i:i+batch_size]
            yb = ys[i:i+batch_size]
            yohb = yohs[i:i+batch_size]

            logits = xb @ W + b
            probs = softmax(logits)

            # gradients
            dlogits = (probs - yohb) / xb.shape[0]                  # (B,K)
            dW = xb.T @ dlogits + l2 * W                             # (D,K)
            db = np.sum(dlogits, axis=0, keepdims=True)              # (1,K)

            W -= lr * dW
            b -= lr * db

        # quick progress
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            pred = np.argmax(X @ W + b, axis=1)
            acc = accuracy(y, pred)
            print(f"epoch {ep:03d} | train acc {acc:.3f}")

    return W, b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"data\folds\tiny_loso_s1.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=r"data\folds\baseline_softmax_s1_model.npz")
    args = ap.parse_args()

    d = np.load(args.data)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int32)
    X_test  = d["X_test"].astype(np.float32)
    y_test  = d["y_test"].astype(np.int32)

    # Map labels to 0..K-1 (stable)
    labels = np.unique(y_train)
    label_to_idx = {int(lbl): i for i, lbl in enumerate(labels)}
    ytr = np.array([label_to_idx[int(v)] for v in y_train], dtype=np.int32)
    yte = np.array([label_to_idx[int(v)] for v in y_test], dtype=np.int32)
    K = len(labels)

    # Features
    Ftr = emg_features(X_train)
    Fte = emg_features(X_test)

    mu, sd = standardize_fit(Ftr)
    Ftr = standardize_apply(Ftr, mu, sd)
    Fte = standardize_apply(Fte, mu, sd)

    # Majority baseline
    maj = np.bincount(ytr).argmax()
    maj_acc = accuracy(yte, np.full_like(yte, maj))
    print(f"Majority-class baseline test acc: {maj_acc:.3f} (K={K})")

    # Train
    W, b = train_softmax_regression(
        Ftr, ytr, K,
        lr=args.lr, l2=args.l2,
        epochs=args.epochs, batch_size=args.batch, seed=args.seed
    )

    # Evaluate
    tr_pred = np.argmax(Ftr @ W + b, axis=1)
    te_pred = np.argmax(Fte @ W + b, axis=1)
    tr_acc = accuracy(ytr, tr_pred)
    te_acc = accuracy(yte, te_pred)
    te_bacc = balanced_accuracy(yte, te_pred, K)

    print("\nFinal:")
    print(f"Train acc: {tr_acc:.3f}")
    print(f"Test  acc: {te_acc:.3f}")
    print(f"Test  balanced acc: {te_bacc:.3f}")

    # Save model + standardization + label map
    np.savez_compressed(
        args.save,
        W=W, b=b, mu=mu, sd=sd, labels=labels
    )
    print("Saved model:", args.save)

if __name__ == "__main__":
    main()
