#!/usr/bin/env python3
import os, glob, json, argparse
import numpy as np
from scipy.io import loadmat

def find_mat_files(subject_dir: str):
    files = sorted(glob.glob(os.path.join(subject_dir, "*.mat*")))
    return [f for f in files if os.path.isfile(f)]

def load_subject_exercises_with_rep(subject_id: str, base_subjects: str):
    folder = os.path.join(base_subjects, subject_id, f"DB3_{subject_id}")
    mat_files = find_mat_files(folder)
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {folder}")

    emg_list, y_list, rep_list = [], [], []
    for fp in mat_files:
        m = loadmat(fp)
        emg_list.append(m["emg"].astype(np.float32))                       # (T, 12)
        y_list.append(m["restimulus"].reshape(-1).astype(np.int32))        # (T,)
        rep_list.append(m["rerepetition"].reshape(-1).astype(np.int32))    # (T,)

    X = np.vstack(emg_list)
    y = np.concatenate(y_list)
    rep = np.concatenate(rep_list)
    return X, y, rep

def contiguous_segments(rep: np.ndarray):
    if rep.ndim != 1:
        rep = rep.reshape(-1)
    cuts = np.where(rep[1:] != rep[:-1])[0] + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(rep)]
    return list(zip(starts, ends))

def window_emg_by_repetition(X, y, rep, fs=2000, win_ms=150, overlap=0.5):
    win_len = int(fs * win_ms / 1000)
    step = int(win_len * (1 - overlap))
    Xw, yw = [], []

    for (a, b) in contiguous_segments(rep):
        if (b - a) < win_len:
            continue
        for start in range(a, b - win_len + 1, step):
            end = start + win_len
            x_win = X[start:end].T
            label = y[start + win_len // 2]
            Xw.append(x_win.astype(np.float32))
            yw.append(int(label))

    return np.stack(Xw), np.array(yw, dtype=np.int32)

def remove_rest(Xw, yw):
    mask = (yw != 0)
    return Xw[mask], yw[mask]

def subsample_per_class(Xw, yw, max_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    keep_idx = []
    for c in np.unique(yw):
        idx = np.where(yw == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    return Xw[keep_idx], yw[keep_idx]

def cap_windows(Xw, yw, max_windows, seed):
    if len(yw) <= max_windows:
        return Xw, yw
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(yw), size=max_windows, replace=False)
    return Xw[idx], yw[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to .../subjects (contains s1, s2, ...)")
    ap.add_argument("--test", default="s1", help="Test subject (default s1)")
    ap.add_argument("--train", default="s2,s3", help="Comma list train subjects (default s2,s3)")
    ap.add_argument("--win_ms", type=int, default=150)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--train_per_class", type=int, default=50)
    ap.add_argument("--test_per_class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/folds/tiny_loso_s1.npz")
    args = ap.parse_args()

    base = args.base
    test_subject = args.test
    train_subjects = [s.strip() for s in args.train.split(",") if s.strip()]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    MAX_WINDOWS = 5000

    # ---- TEST ----
    X, y, rep = load_subject_exercises_with_rep(test_subject, base)
    Xw, yw = window_emg_by_repetition(X, y, rep, win_ms=args.win_ms, overlap=args.overlap)
    Xw, yw = remove_rest(Xw, yw)
    Xw, yw = cap_windows(Xw, yw, MAX_WINDOWS, args.seed)
    X_test, y_test = subsample_per_class(Xw, yw, args.test_per_class, args.seed)

    # ---- TRAIN ----
    X_train_list, y_train_list = [], []
    for sid in train_subjects:
        X, y, rep = load_subject_exercises_with_rep(sid, base)
        Xw, yw = window_emg_by_repetition(X, y, rep, win_ms=args.win_ms, overlap=args.overlap)
        Xw, yw = remove_rest(Xw, yw)
        Xw, yw = cap_windows(Xw, yw, MAX_WINDOWS, args.seed)
        Xw, yw = subsample_per_class(Xw, yw, args.train_per_class, args.seed)
        X_train_list.append(Xw); y_train_list.append(yw)

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    meta = {
        "test_subject": test_subject,
        "train_subjects": train_subjects,
        "win_ms": args.win_ms,
        "overlap": args.overlap,
        "train_per_class": args.train_per_class,
        "test_per_class": args.test_per_class,
        "labels_train": sorted([int(x) for x in np.unique(y_train)]),
        "labels_test": sorted([int(x) for x in np.unique(y_test)]),
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
    }

    np.savez_compressed(args.out, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    with open(args.out.replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", args.out)
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Train classes:", len(np.unique(y_train)), "Test classes:", len(np.unique(y_test)))
    missing = set(np.unique(y_test)) - set(np.unique(y_train))
    print("Classes in TEST not in TRAIN:", sorted([int(x) for x in missing]))

if __name__ == "__main__":
    main()
