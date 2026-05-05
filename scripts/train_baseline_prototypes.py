#!/usr/bin/env python3
import argparse
import numpy as np

def window_norm(X):
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mu) / sd

def emg_features(X, zc_thr=1e-3, ssc_thr=1e-3):
    """
    X: (N,C,T) -> (N, 5C)  [MAV, RMS, WL, ZC, SSC] per channel
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

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

def balanced_accuracy(y_true, y_pred, K):
    recalls = []
    for k in range(K):
        mask = (y_true == k)
        if mask.any():
            recalls.append(np.mean(y_pred[mask] == k))
    return float(np.mean(recalls)) if recalls else 0.0

def l2_normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"data\folds\tiny_loso_s1.npz")
    ap.add_argument("--restrict_to_test", type=int, default=1)
    ap.add_argument("--metric", choices=["cosine","euclid"], default="cosine")
    args = ap.parse_args()

    d = np.load(args.data)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int32)
    X_test  = d["X_test"].astype(np.float32)
    y_test  = d["y_test"].astype(np.int32)

    if args.restrict_to_test == 1:
        test_labels = np.unique(y_test)
        mask = np.isin(y_train, test_labels)
        X_train, y_train = X_train[mask], y_train[mask]
        print(f"Restricted train to test-labels: train classes {len(np.unique(y_train))}; test classes {len(np.unique(y_test))}")

    # map labels to 0..K-1 using TRAIN labels (so test must be subset)
    labels = np.unique(y_train)
    label_to_idx = {int(lbl): i for i, lbl in enumerate(labels)}
    ytr = np.array([label_to_idx[int(v)] for v in y_train], dtype=np.int32)
    yte = np.array([label_to_idx[int(v)] for v in y_test], dtype=np.int32)
    K = len(labels)

    # per-window normalization (helps cross-subject)
    X_train = window_norm(X_train)
    X_test  = window_norm(X_test)

    # features + standardize using train
    Ftr = emg_features(X_train)
    Fte = emg_features(X_test)
    mu, sd = standardize_fit(Ftr)
    Ftr = standardize_apply(Ftr, mu, sd)
    Fte = standardize_apply(Fte, mu, sd)

    # compute class prototypes
    protos = np.zeros((K, Ftr.shape[1]), dtype=np.float32)
    for k in range(K):
        protos[k] = Ftr[ytr == k].mean(axis=0)

    # predict
    if args.metric == "cosine":
        Ftr_n = l2_normalize_rows(Ftr)
        Fte_n = l2_normalize_rows(Fte)
        P_n   = l2_normalize_rows(protos)
        scores = Fte_n @ P_n.T                 # cosine similarity
        ypred = np.argmax(scores, axis=1)
    else:
        # euclidean to prototypes
        # (N,K) distances via (x-p)^2 = x^2 + p^2 - 2xp
        x2 = np.sum(Fte * Fte, axis=1, keepdims=True)
        p2 = np.sum(protos * protos, axis=1, keepdims=True).T
        dist2 = x2 + p2 - 2.0 * (Fte @ protos.T)
        ypred = np.argmin(dist2, axis=1)

    # baselines
    maj = np.bincount(ytr).argmax()
    print(f"Majority baseline test acc: {accuracy(yte, np.full_like(yte, maj)):.3f} (K={K})")

    print("\nPrototype baseline:")
    print(f"Test acc: {accuracy(yte, ypred):.3f}")
    print(f"Test balanced acc: {balanced_accuracy(yte, ypred, K):.3f}")

if __name__ == "__main__":
    main()
