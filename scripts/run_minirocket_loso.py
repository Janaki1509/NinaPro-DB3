<<<<<<< HEAD
#!/usr/bin/env python3
import os, glob, json, argparse, re
import numpy as np
from scipy.io import loadmat

# MiniROCKET + linear classifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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

def contiguous_segments(a: np.ndarray):
    a = a.reshape(-1)
    cuts = np.where(a[1:] != a[:-1])[0] + 1
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, len(a)]
    return list(zip(starts, ends))

def window_by_repetition(X, y, rep, fs=2000, win_ms=150, overlap=0.5, strict_purity=True):
    win_len = int(fs * win_ms / 1000)
    step = int(win_len * (1 - overlap))
    Xw, yw = [], []

    for (a, b) in contiguous_segments(rep):
        if (b - a) < win_len:
            continue
        for s in range(a, b - win_len + 1, step):
            e = s + win_len
            mid = s + win_len // 2
            lbl = int(y[mid])
            if lbl == 0:
                continue
            if strict_purity:
                yy = y[s:e]
                if not (np.all(yy == yy[0]) and int(yy[0]) != 0):
                    continue
            Xw.append(X[s:e].T.astype(np.float32))  # (12,win_len)
            yw.append(lbl)

    if not Xw:
        return np.empty((0, 12, win_len), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.stack(Xw), np.array(yw, dtype=np.int32)

def subsample_per_class(Xw, yw, max_per_class, seed):
    if max_per_class is None:
        return Xw, yw
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(yw):
        idx = np.where(yw == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep) if keep else np.array([], dtype=np.int64)
    rng.shuffle(keep)
    return Xw[keep], yw[keep]

def list_subjects(base_subjects: str):
    subs = []
    for name in os.listdir(base_subjects):
        if os.path.isdir(os.path.join(base_subjects, name)) and re.fullmatch(r"s\d+", name):
            subs.append(name)
    subs = sorted(subs, key=lambda s: int(s[1:]))
    return subs

def compute_std_ratio(X_train, X_test):
    tr_sd = (X_train.std(axis=(0,2)) + 1e-8).mean()
    te_sd = (X_test.std(axis=(0,2)) + 1e-8).mean()
    return float(te_sd / tr_sd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to subjects folder containing s1, s2, ...")
    ap.add_argument("--test", required=True, help="Test subject id, e.g., s1")
    ap.add_argument("--out_json", required=True, help="Where to write per-subject JSON result")
    ap.add_argument("--tmp", default=None, help="Scratch temp dir (used for numba cache)")
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--win_ms", type=int, default=150)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--strict_purity", type=int, default=1, help="1 keeps only pure-label windows (recommended)")
    ap.add_argument("--train_per_class", type=int, default=120)
    ap.add_argument("--test_per_class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_kernels", type=int, default=10000)
    args = ap.parse_args()

    if args.tmp:
        os.environ["NUMBA_CACHE_DIR"] = args.tmp

    strict = (args.strict_purity == 1)

    subjects = list_subjects(args.base)
    if args.test not in subjects:
        raise ValueError(f"Test subject {args.test} not found under base={args.base}")

    train_subjects = [s for s in subjects if s != args.test]

    # ---- build TRAIN ----
    Xtr_list, ytr_list = [], []
    for s in train_subjects:
        X, y, rep = load_subject(s, args.base)
        Xw, yw = window_by_repetition(X, y, rep, win_ms=args.win_ms, overlap=args.overlap, strict_purity=strict)
        Xw, yw = subsample_per_class(Xw, yw, args.train_per_class, args.seed)
        if len(yw) > 0:
            Xtr_list.append(Xw); ytr_list.append(yw)
    X_train = np.concatenate(Xtr_list, axis=0)
    y_train = np.concatenate(ytr_list, axis=0)

    # ---- build TEST ----
    X, y, rep = load_subject(args.test, args.base)
    X_test, y_test = window_by_repetition(X, y, rep, win_ms=args.win_ms, overlap=args.overlap, strict_purity=strict)
    X_test, y_test = subsample_per_class(X_test, y_test, args.test_per_class, args.seed)

    # Restrict train labels to those present in test (fair for held-out subject)
    test_labels = np.unique(y_test)
    m = np.isin(y_train, test_labels)
    X_train, y_train = X_train[m], y_train[m]

    # Map labels -> 0..K-1 using TRAIN labels
    labels = np.unique(y_train)
    l2i = {int(lbl): i for i, lbl in enumerate(labels)}
    ytr = np.array([l2i[int(v)] for v in y_train], dtype=np.int32)
    yte = np.array([l2i[int(v)] for v in y_test], dtype=np.int32)
    K = int(len(labels))

    std_ratio = compute_std_ratio(X_train, X_test)

    # MiniROCKET pipeline (transform + scaler + linear classifier)
    clf = make_pipeline(
        MiniRocketMultivariate(num_kernels=args.num_kernels, n_jobs=args.cpus, random_state=args.seed),
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )

    clf.fit(X_train, ytr)
    pred = clf.predict(X_test)

    acc = float(accuracy_score(yte, pred))
    bacc = float(balanced_accuracy_score(yte, pred))
    macro_f1 = float(f1_score(yte, pred, average="macro"))

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    out = {
        "test_subject": args.test,
        "train_subjects": train_subjects,
        "N_train": int(X_train.shape[0]),
        "N_test": int(X_test.shape[0]),
        "K": K,
        "win_ms": args.win_ms,
        "overlap": args.overlap,
        "strict_purity": strict,
        "train_per_class": args.train_per_class,
        "test_per_class": args.test_per_class,
        "num_kernels": args.num_kernels,
        "std_ratio": std_ratio,
        "acc": acc,
        "balanced_acc": bacc,
        "macro_f1": macro_f1,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{args.test} | K={K} | std_ratio={std_ratio:.3f} | acc={acc:.3f} | bacc={bacc:.3f} | f1={macro_f1:.3f}")
    print("Saved:", args.out_json)

if __name__ == "__main__":
=======
#!/usr/bin/env python3
import os, glob, json, argparse, re
import numpy as np
from scipy.io import loadmat

# MiniROCKET + linear classifier
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

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

def contiguous_segments(a: np.ndarray):
    a = a.reshape(-1)
    cuts = np.where(a[1:] != a[:-1])[0] + 1
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, len(a)]
    return list(zip(starts, ends))

def window_by_repetition(X, y, rep, fs=2000, win_ms=150, overlap=0.5, strict_purity=True):
    win_len = int(fs * win_ms / 1000)
    step = int(win_len * (1 - overlap))
    Xw, yw = [], []

    for (a, b) in contiguous_segments(rep):
        if (b - a) < win_len:
            continue
        for s in range(a, b - win_len + 1, step):
            e = s + win_len
            mid = s + win_len // 2
            lbl = int(y[mid])
            if lbl == 0:
                continue
            if strict_purity:
                yy = y[s:e]
                if not (np.all(yy == yy[0]) and int(yy[0]) != 0):
                    continue
            Xw.append(X[s:e].T.astype(np.float32))  # (12,win_len)
            yw.append(lbl)

    if not Xw:
        return np.empty((0, 12, win_len), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.stack(Xw), np.array(yw, dtype=np.int32)

def subsample_per_class(Xw, yw, max_per_class, seed):
    if max_per_class is None:
        return Xw, yw
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(yw):
        idx = np.where(yw == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep) if keep else np.array([], dtype=np.int64)
    rng.shuffle(keep)
    return Xw[keep], yw[keep]

def list_subjects(base_subjects: str):
    subs = []
    for name in os.listdir(base_subjects):
        if os.path.isdir(os.path.join(base_subjects, name)) and re.fullmatch(r"s\d+", name):
            subs.append(name)
    subs = sorted(subs, key=lambda s: int(s[1:]))
    return subs

def compute_std_ratio(X_train, X_test):
    tr_sd = (X_train.std(axis=(0,2)) + 1e-8).mean()
    te_sd = (X_test.std(axis=(0,2)) + 1e-8).mean()
    return float(te_sd / tr_sd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Path to subjects folder containing s1, s2, ...")
    ap.add_argument("--test", required=True, help="Test subject id, e.g., s1")
    ap.add_argument("--out_json", required=True, help="Where to write per-subject JSON result")
    ap.add_argument("--tmp", default=None, help="Scratch temp dir (used for numba cache)")
    ap.add_argument("--cpus", type=int, default=1)
    ap.add_argument("--win_ms", type=int, default=150)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--strict_purity", type=int, default=1, help="1 keeps only pure-label windows (recommended)")
    ap.add_argument("--train_per_class", type=int, default=120)
    ap.add_argument("--test_per_class", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_kernels", type=int, default=10000)
    args = ap.parse_args()

    if args.tmp:
        os.environ["NUMBA_CACHE_DIR"] = args.tmp

    strict = (args.strict_purity == 1)

    subjects = list_subjects(args.base)
    if args.test not in subjects:
        raise ValueError(f"Test subject {args.test} not found under base={args.base}")

    train_subjects = [s for s in subjects if s != args.test]

    # ---- build TRAIN ----
    Xtr_list, ytr_list = [], []
    for s in train_subjects:
        X, y, rep = load_subject(s, args.base)
        Xw, yw = window_by_repetition(X, y, rep, win_ms=args.win_ms, overlap=args.overlap, strict_purity=strict)
        Xw, yw = subsample_per_class(Xw, yw, args.train_per_class, args.seed)
        if len(yw) > 0:
            Xtr_list.append(Xw); ytr_list.append(yw)
    X_train = np.concatenate(Xtr_list, axis=0)
    y_train = np.concatenate(ytr_list, axis=0)

    # ---- build TEST ----
    X, y, rep = load_subject(args.test, args.base)
    X_test, y_test = window_by_repetition(X, y, rep, win_ms=args.win_ms, overlap=args.overlap, strict_purity=strict)
    X_test, y_test = subsample_per_class(X_test, y_test, args.test_per_class, args.seed)

    # Restrict train labels to those present in test (fair for held-out subject)
    test_labels = np.unique(y_test)
    m = np.isin(y_train, test_labels)
    X_train, y_train = X_train[m], y_train[m]

    # Map labels -> 0..K-1 using TRAIN labels
    labels = np.unique(y_train)
    l2i = {int(lbl): i for i, lbl in enumerate(labels)}
    ytr = np.array([l2i[int(v)] for v in y_train], dtype=np.int32)
    yte = np.array([l2i[int(v)] for v in y_test], dtype=np.int32)
    K = int(len(labels))

    std_ratio = compute_std_ratio(X_train, X_test)

    # MiniROCKET pipeline (transform + scaler + linear classifier)
    clf = make_pipeline(
        MiniRocketMultivariate(num_kernels=args.num_kernels, n_jobs=args.cpus, random_state=args.seed),
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )

    clf.fit(X_train, ytr)
    pred = clf.predict(X_test)

    acc = float(accuracy_score(yte, pred))
    bacc = float(balanced_accuracy_score(yte, pred))
    macro_f1 = float(f1_score(yte, pred, average="macro"))

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    out = {
        "test_subject": args.test,
        "train_subjects": train_subjects,
        "N_train": int(X_train.shape[0]),
        "N_test": int(X_test.shape[0]),
        "K": K,
        "win_ms": args.win_ms,
        "overlap": args.overlap,
        "strict_purity": strict,
        "train_per_class": args.train_per_class,
        "test_per_class": args.test_per_class,
        "num_kernels": args.num_kernels,
        "std_ratio": std_ratio,
        "acc": acc,
        "balanced_acc": bacc,
        "macro_f1": macro_f1,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{args.test} | K={K} | std_ratio={std_ratio:.3f} | acc={acc:.3f} | bacc={bacc:.3f} | f1={macro_f1:.3f}")
    print("Saved:", args.out_json)

if __name__ == "__main__":
>>>>>>> 9bbfa3b24bb49262e519b5672b0b6636e2cc4682
    main()