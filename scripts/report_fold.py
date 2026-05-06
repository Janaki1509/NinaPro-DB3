<<<<<<< HEAD
#!/usr/bin/env python3
import argparse
import numpy as np

# ---------------- basics ----------------
def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def balanced_accuracy(y_true, y_pred, K):
    rec = []
    for k in range(K):
        m = (y_true == k)
        if m.any():
            rec.append(float(np.mean(y_pred[m] == k)))
    return float(np.mean(rec)) if rec else 0.0

def window_norm(X):
    # per-window per-channel normalization across time
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mu) / sd

def emg_features(X, zc_thr=1e-3, ssc_thr=1e-3):
    """
    X: (N,C,T) -> (N,5C) : MAV, RMS, WL, ZC, SSC per channel
    """
    eps = 1e-8
    mav = np.mean(np.abs(X), axis=2)
    rms = np.sqrt(np.mean(X * X, axis=2) + eps)
    wl  = np.sum(np.abs(np.diff(X, axis=2)), axis=2)

    x1 = X[:, :, :-1]
    x2 = X[:, :,  1:]
    zc = np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) > zc_thr), axis=2).astype(np.float32)

    d1 = X[:, :, 1:-1] - X[:, :, :-2]
    d2 = X[:, :, 2:  ] - X[:, :, 1:-1]
    ssc = np.sum(((d1 * d2) < 0) & ((np.abs(d1) > ssc_thr) | (np.abs(d2) > ssc_thr)), axis=2).astype(np.float32)

    return np.concatenate([mav, rms, wl, zc, ssc], axis=1).astype(np.float32)

def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def l2_normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / n

# ---------------- softmax regression ----------------
def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def one_hot(y, K):
    oh = np.zeros((y.size, K), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

def train_softmax(X, y, K, lr=0.05, l2=1e-3, epochs=60, batch=256, seed=42):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    W = 0.01 * rng.standard_normal((D, K)).astype(np.float32)
    b = np.zeros((1, K), dtype=np.float32)
    Y = one_hot(y, K)

    for ep in range(1, epochs + 1):
        idx = rng.permutation(N)
        Xs, Ys = X[idx], Y[idx]
        for i in range(0, N, batch):
            xb = Xs[i:i+batch]
            yb = Ys[i:i+batch]
            p = softmax(xb @ W + b)
            dlogits = (p - yb) / xb.shape[0]
            dW = xb.T @ dlogits + l2 * W
            db = np.sum(dlogits, axis=0, keepdims=True)
            W -= lr * dW
            b -= lr * db

        if ep in (1, 10, 20, 40, epochs):
            pred = np.argmax(X @ W + b, axis=1)
            print(f"  softmax ep {ep:03d} train acc {accuracy(y, pred):.3f}")

    return W, b

# ---------------- report ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"data\folds\tiny_loso_s1.npz")
    ap.add_argument("--restrict_to_test", type=int, default=1, help="1 recommended for LOSO tiny")
    ap.add_argument("--use_window_norm", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.data)
    Xtr = d["X_train"].astype(np.float32)  # (N,C,T)
    ytr_raw = d["y_train"].astype(np.int32)
    Xte = d["X_test"].astype(np.float32)
    yte_raw = d["y_test"].astype(np.int32)

    print("=== FOLD ===")
    print("data:", args.data)
    print("X_train:", Xtr.shape, "X_test:", Xte.shape)
    print("train labels:", len(np.unique(ytr_raw)), "test labels:", len(np.unique(yte_raw)))

    # invariant checks
    print("\n=== INVARIANTS ===")
    print("NaNs in X_train:", int(np.isnan(Xtr).sum()), "Infs:", int(np.isinf(Xtr).sum()))
    print("NaNs in X_test :", int(np.isnan(Xte).sum()), "Infs:", int(np.isinf(Xte).sum()))
    print("abs mean train:", float(np.mean(np.abs(Xtr))), "abs mean test:", float(np.mean(np.abs(Xte))))
    # per-channel stats (raw)
    tr_mu_c = Xtr.mean(axis=(0,2))
    te_mu_c = Xte.mean(axis=(0,2))
    tr_sd_c = Xtr.std(axis=(0,2)) + 1e-8
    te_sd_c = Xte.std(axis=(0,2)) + 1e-8
    print("per-channel mean delta (test-train) L1:", float(np.sum(np.abs(te_mu_c - tr_mu_c))))
    print("per-channel std  ratio (test/train)  mean:", float(np.mean(te_sd_c / tr_sd_c)))

    train_labels = np.unique(ytr_raw)
    test_labels  = np.unique(yte_raw)
    missing = set(map(int, test_labels)) - set(map(int, train_labels))
    extra   = set(map(int, train_labels)) - set(map(int, test_labels))
    print("labels in test not in train:", sorted(list(missing)))
    print("labels in train not in test:", sorted(list(extra))[:20], ("...(+"+str(max(0,len(extra)-20))+")" if len(extra)>20 else ""))

    # restrict train to test labels (important for tiny LOSO)
    if args.restrict_to_test == 1:
        m = np.isin(ytr_raw, test_labels)
        Xtr = Xtr[m]
        ytr_raw = ytr_raw[m]
        print("\nRestricted train to test-labels -> train N:", len(ytr_raw), "classes:", len(np.unique(ytr_raw)))

    # map labels to 0..K-1 using TRAIN labels
    labels = np.unique(ytr_raw)
    l2i = {int(lbl): i for i, lbl in enumerate(labels)}
    ytr = np.array([l2i[int(v)] for v in ytr_raw], dtype=np.int32)
    yte = np.array([l2i[int(v)] for v in yte_raw], dtype=np.int32)
    K = len(labels)

    print("\n=== MAJORITY BASELINE ===")
    maj = np.bincount(ytr).argmax()
    ypred = np.full_like(yte, maj)
    print("test acc:", f"{accuracy(yte, ypred):.3f}", "test bacc:", f"{balanced_accuracy(yte, ypred, K):.3f}", "K:", K)

    # feature pipeline
    if args.use_window_norm == 1:
        Xtr2 = window_norm(Xtr)
        Xte2 = window_norm(Xte)
    else:
        Xtr2, Xte2 = Xtr, Xte

    Ftr = emg_features(Xtr2)
    Fte = emg_features(Xte2)
    mu, sd = standardize_fit(Ftr)
    Ftr = standardize_apply(Ftr, mu, sd)
    Fte = standardize_apply(Fte, mu, sd)

    print("\n=== PROTOTYPE BASELINE ===")
    protos = np.zeros((K, Ftr.shape[1]), dtype=np.float32)
    for k in range(K):
        protos[k] = Ftr[ytr == k].mean(axis=0)

    # cosine
    Fte_n = l2_normalize_rows(Fte)
    P_n   = l2_normalize_rows(protos)
    scores = Fte_n @ P_n.T
    pred_cos = np.argmax(scores, axis=1)
    print("cosine  test acc:", f"{accuracy(yte, pred_cos):.3f}",
          "bacc:", f"{balanced_accuracy(yte, pred_cos, K):.3f}")

    # euclid
    x2 = np.sum(Fte * Fte, axis=1, keepdims=True)
    p2 = np.sum(protos * protos, axis=1, keepdims=True).T
    dist2 = x2 + p2 - 2.0 * (Fte @ protos.T)
    pred_euc = np.argmin(dist2, axis=1)
    print("euclid  test acc:", f"{accuracy(yte, pred_euc):.3f}",
          "bacc:", f"{balanced_accuracy(yte, pred_euc, K):.3f}")

    print("\n=== EMG FEATURES + SOFTMAX ===")
    W, b = train_softmax(Ftr, ytr, K, lr=args.lr, l2=args.l2, epochs=args.epochs, seed=args.seed)
    pred = np.argmax(Fte @ W + b, axis=1)
    print("test acc:", f"{accuracy(yte, pred):.3f}", "test bacc:", f"{balanced_accuracy(yte, pred, K):.3f}")

if __name__ == "__main__":
    main()
=======
#!/usr/bin/env python3
import argparse
import numpy as np

# ---------------- basics ----------------
def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def balanced_accuracy(y_true, y_pred, K):
    rec = []
    for k in range(K):
        m = (y_true == k)
        if m.any():
            rec.append(float(np.mean(y_pred[m] == k)))
    return float(np.mean(rec)) if rec else 0.0

def window_norm(X):
    # per-window per-channel normalization across time
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mu) / sd

def emg_features(X, zc_thr=1e-3, ssc_thr=1e-3):
    """
    X: (N,C,T) -> (N,5C) : MAV, RMS, WL, ZC, SSC per channel
    """
    eps = 1e-8
    mav = np.mean(np.abs(X), axis=2)
    rms = np.sqrt(np.mean(X * X, axis=2) + eps)
    wl  = np.sum(np.abs(np.diff(X, axis=2)), axis=2)

    x1 = X[:, :, :-1]
    x2 = X[:, :,  1:]
    zc = np.sum(((x1 * x2) < 0) & (np.abs(x1 - x2) > zc_thr), axis=2).astype(np.float32)

    d1 = X[:, :, 1:-1] - X[:, :, :-2]
    d2 = X[:, :, 2:  ] - X[:, :, 1:-1]
    ssc = np.sum(((d1 * d2) < 0) & ((np.abs(d1) > ssc_thr) | (np.abs(d2) > ssc_thr)), axis=2).astype(np.float32)

    return np.concatenate([mav, rms, wl, zc, ssc], axis=1).astype(np.float32)

def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def standardize_apply(X, mu, sd):
    return (X - mu) / sd

def l2_normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / n

# ---------------- softmax regression ----------------
def softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def one_hot(y, K):
    oh = np.zeros((y.size, K), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh

def train_softmax(X, y, K, lr=0.05, l2=1e-3, epochs=60, batch=256, seed=42):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    W = 0.01 * rng.standard_normal((D, K)).astype(np.float32)
    b = np.zeros((1, K), dtype=np.float32)
    Y = one_hot(y, K)

    for ep in range(1, epochs + 1):
        idx = rng.permutation(N)
        Xs, Ys = X[idx], Y[idx]
        for i in range(0, N, batch):
            xb = Xs[i:i+batch]
            yb = Ys[i:i+batch]
            p = softmax(xb @ W + b)
            dlogits = (p - yb) / xb.shape[0]
            dW = xb.T @ dlogits + l2 * W
            db = np.sum(dlogits, axis=0, keepdims=True)
            W -= lr * dW
            b -= lr * db

        if ep in (1, 10, 20, 40, epochs):
            pred = np.argmax(X @ W + b, axis=1)
            print(f"  softmax ep {ep:03d} train acc {accuracy(y, pred):.3f}")

    return W, b

# ---------------- report ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"data\folds\tiny_loso_s1.npz")
    ap.add_argument("--restrict_to_test", type=int, default=1, help="1 recommended for LOSO tiny")
    ap.add_argument("--use_window_norm", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.data)
    Xtr = d["X_train"].astype(np.float32)  # (N,C,T)
    ytr_raw = d["y_train"].astype(np.int32)
    Xte = d["X_test"].astype(np.float32)
    yte_raw = d["y_test"].astype(np.int32)

    print("=== FOLD ===")
    print("data:", args.data)
    print("X_train:", Xtr.shape, "X_test:", Xte.shape)
    print("train labels:", len(np.unique(ytr_raw)), "test labels:", len(np.unique(yte_raw)))

    # invariant checks
    print("\n=== INVARIANTS ===")
    print("NaNs in X_train:", int(np.isnan(Xtr).sum()), "Infs:", int(np.isinf(Xtr).sum()))
    print("NaNs in X_test :", int(np.isnan(Xte).sum()), "Infs:", int(np.isinf(Xte).sum()))
    print("abs mean train:", float(np.mean(np.abs(Xtr))), "abs mean test:", float(np.mean(np.abs(Xte))))
    # per-channel stats (raw)
    tr_mu_c = Xtr.mean(axis=(0,2))
    te_mu_c = Xte.mean(axis=(0,2))
    tr_sd_c = Xtr.std(axis=(0,2)) + 1e-8
    te_sd_c = Xte.std(axis=(0,2)) + 1e-8
    print("per-channel mean delta (test-train) L1:", float(np.sum(np.abs(te_mu_c - tr_mu_c))))
    print("per-channel std  ratio (test/train)  mean:", float(np.mean(te_sd_c / tr_sd_c)))

    train_labels = np.unique(ytr_raw)
    test_labels  = np.unique(yte_raw)
    missing = set(map(int, test_labels)) - set(map(int, train_labels))
    extra   = set(map(int, train_labels)) - set(map(int, test_labels))
    print("labels in test not in train:", sorted(list(missing)))
    print("labels in train not in test:", sorted(list(extra))[:20], ("...(+"+str(max(0,len(extra)-20))+")" if len(extra)>20 else ""))

    # restrict train to test labels (important for tiny LOSO)
    if args.restrict_to_test == 1:
        m = np.isin(ytr_raw, test_labels)
        Xtr = Xtr[m]
        ytr_raw = ytr_raw[m]
        print("\nRestricted train to test-labels -> train N:", len(ytr_raw), "classes:", len(np.unique(ytr_raw)))

    # map labels to 0..K-1 using TRAIN labels
    labels = np.unique(ytr_raw)
    l2i = {int(lbl): i for i, lbl in enumerate(labels)}
    ytr = np.array([l2i[int(v)] for v in ytr_raw], dtype=np.int32)
    yte = np.array([l2i[int(v)] for v in yte_raw], dtype=np.int32)
    K = len(labels)

    print("\n=== MAJORITY BASELINE ===")
    maj = np.bincount(ytr).argmax()
    ypred = np.full_like(yte, maj)
    print("test acc:", f"{accuracy(yte, ypred):.3f}", "test bacc:", f"{balanced_accuracy(yte, ypred, K):.3f}", "K:", K)

    # feature pipeline
    if args.use_window_norm == 1:
        Xtr2 = window_norm(Xtr)
        Xte2 = window_norm(Xte)
    else:
        Xtr2, Xte2 = Xtr, Xte

    Ftr = emg_features(Xtr2)
    Fte = emg_features(Xte2)
    mu, sd = standardize_fit(Ftr)
    Ftr = standardize_apply(Ftr, mu, sd)
    Fte = standardize_apply(Fte, mu, sd)

    print("\n=== PROTOTYPE BASELINE ===")
    protos = np.zeros((K, Ftr.shape[1]), dtype=np.float32)
    for k in range(K):
        protos[k] = Ftr[ytr == k].mean(axis=0)

    # cosine
    Fte_n = l2_normalize_rows(Fte)
    P_n   = l2_normalize_rows(protos)
    scores = Fte_n @ P_n.T
    pred_cos = np.argmax(scores, axis=1)
    print("cosine  test acc:", f"{accuracy(yte, pred_cos):.3f}",
          "bacc:", f"{balanced_accuracy(yte, pred_cos, K):.3f}")

    # euclid
    x2 = np.sum(Fte * Fte, axis=1, keepdims=True)
    p2 = np.sum(protos * protos, axis=1, keepdims=True).T
    dist2 = x2 + p2 - 2.0 * (Fte @ protos.T)
    pred_euc = np.argmin(dist2, axis=1)
    print("euclid  test acc:", f"{accuracy(yte, pred_euc):.3f}",
          "bacc:", f"{balanced_accuracy(yte, pred_euc, K):.3f}")

    print("\n=== EMG FEATURES + SOFTMAX ===")
    W, b = train_softmax(Ftr, ytr, K, lr=args.lr, l2=args.l2, epochs=args.epochs, seed=args.seed)
    pred = np.argmax(Fte @ W + b, axis=1)
    print("test acc:", f"{accuracy(yte, pred):.3f}", "test bacc:", f"{balanced_accuracy(yte, pred, K):.3f}")

if __name__ == "__main__":
    main()
>>>>>>> 9bbfa3b24bb49262e519b5672b0b6636e2cc4682
