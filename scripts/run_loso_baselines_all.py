#!/usr/bin/env python3
import os, re, argparse, subprocess, sys
import numpy as np

# --- tiny utilities (copied minimal from your report_fold logic) ---
def accuracy(y, p): return float(np.mean(y == p))

def balanced_accuracy(y, p, K):
    rec=[]
    for k in range(K):
        m=(y==k)
        if m.any(): rec.append(float(np.mean(p[m]==k)))
    return float(np.mean(rec)) if rec else 0.0

def window_norm(X):
    mu = X.mean(axis=2, keepdims=True)
    sd = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mu) / sd

def emg_features(X, zc_thr=1e-3, ssc_thr=1e-3):
    eps=1e-8
    mav=np.mean(np.abs(X), axis=2)
    rms=np.sqrt(np.mean(X*X, axis=2)+eps)
    wl =np.sum(np.abs(np.diff(X, axis=2)), axis=2)

    x1=X[:,:,:-1]; x2=X[:,:,1:]
    zc=np.sum(((x1*x2)<0) & (np.abs(x1-x2)>zc_thr), axis=2).astype(np.float32)

    d1=X[:,:,1:-1]-X[:,:,:-2]
    d2=X[:,:,2:  ]-X[:,:,1:-1]
    ssc=np.sum(((d1*d2)<0) & ((np.abs(d1)>ssc_thr)|(np.abs(d2)>ssc_thr)), axis=2).astype(np.float32)

    return np.concatenate([mav,rms,wl,zc,ssc], axis=1).astype(np.float32)

def standardize_fit(X):
    mu=X.mean(axis=0, keepdims=True)
    sd=X.std(axis=0, keepdims=True)+1e-8
    return mu, sd

def standardize_apply(X, mu, sd): return (X-mu)/sd

def l2_norm_rows(X):
    n=np.linalg.norm(X, axis=1, keepdims=True)+1e-8
    return X/n

def softmax(z):
    z=z-np.max(z, axis=1, keepdims=True)
    e=np.exp(z)
    return e/np.sum(e, axis=1, keepdims=True)

def one_hot(y, K):
    oh=np.zeros((y.size, K), dtype=np.float32)
    oh[np.arange(y.size), y]=1.0
    return oh

def train_softmax_reg(Ftr, ytr, K, lr=0.05, l2=1e-3, epochs=30, batch=512, seed=42):
    rng=np.random.default_rng(seed)
    N,D=Ftr.shape
    W=(0.01*rng.standard_normal((D,K))).astype(np.float32)
    b=np.zeros((1,K), dtype=np.float32)
    Y=one_hot(ytr, K)
    for _ in range(epochs):
        idx=rng.permutation(N)
        Xs, Ys = Ftr[idx], Y[idx]
        for i in range(0, N, batch):
            xb=Js = Xs[i:i+batch]
            yb=Ys[i:i+batch]
            p=softmax(xb@W+b)
            d=(p-yb)/xb.shape[0]
            W -= lr*(xb.T@d + l2*W)
            b -= lr*(np.sum(d, axis=0, keepdims=True))
    return W,b

def list_subjects(base):
    subs=[]
    for name in os.listdir(base):
        if os.path.isdir(os.path.join(base,name)) and re.fullmatch(r"s\d+", name):
            subs.append(name)
    return sorted(subs, key=lambda s:int(s[1:]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", default=r"C:\ninapro_db3\subjects")
    ap.add_argument("--train_per_class", type=int, default=120)
    ap.add_argument("--test_per_class", type=int, default=80)
    ap.add_argument("--out_csv", default=r"results_loso_baselines.csv")
    ap.add_argument("--keep_folds", type=int, default=0)
    args=ap.parse_args()

    subs=list_subjects(args.base)
    if len(subs)<2:
        raise SystemExit("Not enough subjects found under: "+args.base)

    rows=[]
    for test in subs:
        train=[s for s in subs if s!=test]
        fold_path = os.path.join("data","folds",f"loso_{test}_cap.npz")
        os.makedirs(os.path.dirname(fold_path), exist_ok=True)

        # generate fold
        cmd=[sys.executable, "make_tiny_loso.py",
             "--base", args.base,
             "--test", test,
             "--train", ",".join(train),
             "--train_per_class", str(args.train_per_class),
             "--test_per_class", str(args.test_per_class),
             "--out", fold_path]
        subprocess.check_call(cmd)

        d=np.load(fold_path)
        Xtr=d["X_train"].astype(np.float32); ytr_raw=d["y_train"].astype(np.int32)
        Xte=d["X_test"].astype(np.float32);  yte_raw=d["y_test"].astype(np.int32)

        # invariants
        abs_mean_tr=float(np.mean(np.abs(Xtr)))
        abs_mean_te=float(np.mean(np.abs(Xte)))
        tr_sd=float(np.mean((Xtr.std(axis=(0,2))+1e-8)))
        te_sd=float(np.mean((Xte.std(axis=(0,2))+1e-8)))
        std_ratio=float(te_sd/tr_sd)

        # restrict train to test labels
        test_labels=np.unique(yte_raw)
        m=np.isin(ytr_raw, test_labels)
        Xtr=Xtr[m]; ytr_raw=ytr_raw[m]

        labels=np.unique(ytr_raw)
        l2i={int(lbl):i for i,lbl in enumerate(labels)}
        ytr=np.array([l2i[int(v)] for v in ytr_raw], dtype=np.int32)
        yte=np.array([l2i[int(v)] for v in yte_raw], dtype=np.int32)
        K=len(labels)

        # majority
        maj=np.bincount(ytr).argmax()
        pred=np.full_like(yte, maj)
        maj_acc=accuracy(yte, pred); maj_bacc=balanced_accuracy(yte, pred, K)

        # features
        Xtr2=window_norm(Xtr); Xte2=window_norm(Xte)
        Ftr=emg_features(Xtr2); Fte=emg_features(Xte2)
        mu,sd=standardize_fit(Ftr)
        Ftr=standardize_apply(Ftr, mu, sd); Fte=standardize_apply(Fte, mu, sd)

        # prototype cosine
        protos=np.zeros((K,Ftr.shape[1]), dtype=np.float32)
        for k in range(K): protos[k]=Ftr[ytr==k].mean(axis=0)
        scores=l2_norm_rows(Fte) @ l2_norm_rows(protos).T
        pred=np.argmax(scores, axis=1)
        proto_acc=accuracy(yte, pred); proto_bacc=balanced_accuracy(yte, pred, K)

        # softmax reg
        W,b=train_softmax_reg(Ftr, ytr, K, epochs=30)
        pred=np.argmax(Fte@W+b, axis=1)
        soft_acc=accuracy(yte, pred); soft_bacc=balanced_accuracy(yte, pred, K)

        rows.append([test, len(Xtr), len(Xte), K,
                     abs_mean_tr, abs_mean_te, std_ratio,
                     maj_acc, maj_bacc,
                     proto_acc, proto_bacc,
                     soft_acc, soft_bacc])

        print(f"{test}: K={K} std_ratio={std_ratio:.3f} maj={maj_acc:.3f} proto={proto_acc:.3f} soft={soft_acc:.3f}")

        if args.keep_folds==0:
            try: os.remove(fold_path)
            except OSError: pass

    # write CSV
    header=("test_subject,train_N,test_N,K,absmean_train,absmean_test,std_ratio,"
            "maj_acc,maj_bacc,proto_acc,proto_bacc,soft_acc,soft_bacc\n")
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(header)
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    print("Saved:", args.out_csv)

if __name__=="__main__":
    main()
