#!/usr/bin/env python3
import os, glob, json, argparse
import numpy as np
from scipy.io import loadmat

def find_mat_files(subject_dir: str):
    files = sorted(glob.glob(os.path.join(subject_dir, "*.mat*")))
    return [f for f in files if os.path.isfile(f)]

def load_subject(subject_id: str, base_subjects: str):
    folder = os.path.join(base_subjects, subject_id, f"DB3_{subject_id}")
    mats = find_mat_files(folder)
    if not mats:
        raise FileNotFoundError(f"No .mat files found in: {folder}")

    emg_list, y_list, rep_list = [], [], []
    for fp in mats:
        m = loadmat(fp)
        emg_list.append(m["emg"].astype(np.float32))                       # (T,12)
        y_list.append(m["restimulus"].reshape(-1).astype(np.int32))        # (T,)
        rep_list.append(m["rerepetition"].reshape(-1).astype(np.int32))    # (T,)
    X = np.vstack(emg_list)
    y = np.concatenate(y_list)
    rep = np.concatenate(rep_list)
    return X, y, rep

def contiguous_segments(rep: np.ndarray):
    rep = rep.reshape(-1)
    cuts = np.where(rep[1:] != rep[:-1])[0] + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(rep)]
    return list(zip(starts, ends))

def window_by_rep(X, y, rep, fs=2000, win_ms=150, overlap=0.5):
    win_len = int(fs * win_ms / 1000)
    step = int(win_len * (1 - overlap))
    Xw, yw, rw = [], [], []

    for (a, b) in contiguous_segments(rep):
        if (b - a) < win_len:
            continue
        for start in range(a, b - win_len + 1, step):
            end = start + win_len
            mid = start + win_len // 2
            Xw.append(X[start:end].T.astype(np.float32))  # (12,win_len)
            yw.append(int(y[mid]))
            rw.append(int(rep[mid]))
    return np.stack(Xw), np.array(yw, np.int32), np.array(rw, np.int32)

def remove_rest(Xw, yw, rw):
    m = (yw != 0)
    return Xw[m], yw[m], rw[m]

def subsample_per_class(Xw, yw, max_per_class, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(yw):
        idx = np.where(yw == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return Xw[keep], yw[keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--subject", default="s1")
    ap.add_argument("--train_reps", default="1,2,3,4")
    ap.add_argument("--test_reps", default="5,6")
    ap.add_argument("--win_ms", type=int, default=150)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--train_per_class", type=int, default=300)
    ap.add_argument("--test_per_class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=r"data\folds\within_s1_rep.npz")
    args = ap.parse_args()

    tr_reps = set(int(x) for x in args.train_reps.split(",") if x.strip())
    te_reps = set(int(x) for x in args.test_reps.split(",") if x.strip())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    X, y, rep = load_subject(args.subject, args.base)
    Xw, yw, rw = window_by_rep(X, y, rep, win_ms=args.win_ms, overlap=args.overlap)
    Xw, yw, rw = remove_rest(Xw, yw, rw)

    mtr = np.isin(rw, list(tr_reps))
    mte = np.isin(rw, list(te_reps))
    Xtr, ytr = Xw[mtr], yw[mtr]
    Xte, yte = Xw[mte], yw[mte]

    # keep only labels present in both
    common = np.intersect1d(np.unique(ytr), np.unique(yte))
    Xtr, ytr = Xtr[np.isin(ytr, common)], ytr[np.isin(ytr, common)]
    Xte, yte = Xte[np.isin(yte, common)], yte[np.isin(yte, common)]

    Xtr, ytr = subsample_per_class(Xtr, ytr, args.train_per_class, args.seed)
    Xte, yte = subsample_per_class(Xte, yte, args.test_per_class, args.seed)

    np.savez_compressed(args.out, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)

    meta = {
        "subject": args.subject,
        "train_reps": sorted(list(tr_reps)),
        "test_reps": sorted(list(te_reps)),
        "win_ms": args.win_ms,
        "overlap": args.overlap,
        "train_per_class": args.train_per_class,
        "test_per_class": args.test_per_class,
        "X_train_shape": list(Xtr.shape),
        "X_test_shape": list(Xte.shape),
        "classes": int(len(np.unique(ytr)))
    }
    with open(args.out.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", args.out)
    print("Train:", Xtr.shape, "Test:", Xte.shape, "Classes:", len(np.unique(ytr)))

if __name__ == "__main__":
    main()
