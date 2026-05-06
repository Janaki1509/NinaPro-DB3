<<<<<<< HEAD
#!/usr/bin/env python3
import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
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

def contiguous_segments(a: np.ndarray):
    a = a.reshape(-1)
    cuts = np.where(a[1:] != a[:-1])[0] + 1
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, len(a)]
    return list(zip(starts, ends))

def window_by_repetition_with_indices(X, y, rep, fs=2000, win_ms=150, overlap=0.5):
    win_len = int(fs * win_ms / 1000)
    step = int(win_len * (1 - overlap))
    starts, ends, labels = [], [], []
    for (a, b) in contiguous_segments(rep):
        if (b - a) < win_len:
            continue
        for s in range(a, b - win_len + 1, step):
            e = s + win_len
            lbl = int(y[s + win_len // 2])  # center label
            starts.append(s); ends.append(e); labels.append(lbl)
    return np.array(starts), np.array(ends), np.array(labels, dtype=np.int32)

def class_counts(y):
    ys, cs = np.unique(y, return_counts=True)
    return ys.astype(int), cs.astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="Path to fold .npz")
    ap.add_argument("--base", required=True, help="Path to ...\\subjects")
    ap.add_argument("--test_subject", required=True, help="e.g., s1")
    ap.add_argument("--win_ms", type=int, default=150)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--outdir", default=r"outputs\one_fold")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Fold info + tensor shape ---
    d = np.load(args.fold)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int32)
    X_test  = d["X_test"].astype(np.float32)
    y_test  = d["y_test"].astype(np.int32)

    print("=== Example input tensor shape ===")
    print("X_train shape:", X_train.shape, "(N,C,T)")
    print("One sample shape:", X_train[0].shape, "(C,T)")
    print("LSTM-style input shape would be:", (X_train[0].shape[1], X_train[0].shape[0]), "(T,C)")

    # --- Class histogram train vs test ---
    yt, ct = class_counts(y_train)
    ye, ce = class_counts(y_test)
    all_labels = np.unique(np.concatenate([yt, ye]))
    train_map = {k:v for k,v in zip(yt, ct)}
    test_map  = {k:v for k,v in zip(ye, ce)}
    train_counts = np.array([train_map.get(int(k), 0) for k in all_labels])
    test_counts  = np.array([test_map.get(int(k), 0) for k in all_labels])

    x = np.arange(len(all_labels))
    plt.figure(figsize=(10,4))
    plt.bar(x - 0.2, train_counts, width=0.4, label="Train")
    plt.bar(x + 0.2, test_counts,  width=0.4, label="Test")
    plt.xticks(x, all_labels, rotation=90)
    plt.ylabel("Count")
    plt.title("Class Histogram: Train vs Test")
    plt.legend()
    plt.tight_layout()
    p_hist = os.path.join(args.outdir, "class_hist_train_vs_test.png")
    plt.savefig(p_hist, dpi=200)
    plt.close()
    print("Saved:", p_hist)

    # --- Window-within-gesture-interval check (on RAW test subject) ---
    # We verify if each window lies entirely inside one constant restimulus interval.
    Xraw, yraw, rep = load_subject(args.test_subject, args.base)
    s_idx, e_idx, y_center = window_by_repetition_with_indices(
        Xraw, yraw, rep, win_ms=args.win_ms, overlap=args.overlap
    )

    # ignore rest (0) like your pipeline
    keep = (y_center != 0)
    s_idx, e_idx, y_center = s_idx[keep], e_idx[keep], y_center[keep]

    # check if y is constant inside each window
    ok = 0
    for s, e in zip(s_idx, e_idx):
        yy = yraw[s:e]
        if np.all(yy == yy[0]) and yy[0] != 0:
            ok += 1
    frac_ok = ok / len(s_idx) if len(s_idx) else 0.0

    print("\n=== Window boundary check (test subject raw) ===")
    print("Subject:", args.test_subject)
    print("Windows checked (non-rest):", int(len(s_idx)))
    print("Fraction fully inside a single gesture interval:", float(frac_ok))

    # save a small text summary
    p_txt = os.path.join(args.outdir, "window_interval_check.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write(f"subject={args.test_subject}\n")
        f.write(f"windows_non_rest={int(len(s_idx))}\n")
        f.write(f"fraction_inside_single_gesture_interval={float(frac_ok)}\n")
    print("Saved:", p_txt)

if __name__ == "__main__":
    main()
=======
#!/usr/bin/env python3
import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
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

def contiguous_segments(a: np.ndarray):
    a = a.reshape(-1)
    cuts = np.where(a[1:] != a[:-1])[0] + 1
    starts = np.r_[0, cuts]
    ends   = np.r_[cuts, len(a)]
    return list(zip(starts, ends))

def window_by_repetition_with_indices(X, y, rep, fs=2000, win_ms=150, overlap=0.5):
    win_len = int(fs * win_ms / 1000)
    step = int(win_len * (1 - overlap))
    starts, ends, labels = [], [], []
    for (a, b) in contiguous_segments(rep):
        if (b - a) < win_len:
            continue
        for s in range(a, b - win_len + 1, step):
            e = s + win_len
            lbl = int(y[s + win_len // 2])  # center label
            starts.append(s); ends.append(e); labels.append(lbl)
    return np.array(starts), np.array(ends), np.array(labels, dtype=np.int32)

def class_counts(y):
    ys, cs = np.unique(y, return_counts=True)
    return ys.astype(int), cs.astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="Path to fold .npz")
    ap.add_argument("--base", required=True, help="Path to ...\\subjects")
    ap.add_argument("--test_subject", required=True, help="e.g., s1")
    ap.add_argument("--win_ms", type=int, default=150)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--outdir", default=r"outputs\one_fold")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- Fold info + tensor shape ---
    d = np.load(args.fold)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int32)
    X_test  = d["X_test"].astype(np.float32)
    y_test  = d["y_test"].astype(np.int32)

    print("=== Example input tensor shape ===")
    print("X_train shape:", X_train.shape, "(N,C,T)")
    print("One sample shape:", X_train[0].shape, "(C,T)")
    print("LSTM-style input shape would be:", (X_train[0].shape[1], X_train[0].shape[0]), "(T,C)")

    # --- Class histogram train vs test ---
    yt, ct = class_counts(y_train)
    ye, ce = class_counts(y_test)
    all_labels = np.unique(np.concatenate([yt, ye]))
    train_map = {k:v for k,v in zip(yt, ct)}
    test_map  = {k:v for k,v in zip(ye, ce)}
    train_counts = np.array([train_map.get(int(k), 0) for k in all_labels])
    test_counts  = np.array([test_map.get(int(k), 0) for k in all_labels])

    x = np.arange(len(all_labels))
    plt.figure(figsize=(10,4))
    plt.bar(x - 0.2, train_counts, width=0.4, label="Train")
    plt.bar(x + 0.2, test_counts,  width=0.4, label="Test")
    plt.xticks(x, all_labels, rotation=90)
    plt.ylabel("Count")
    plt.title("Class Histogram: Train vs Test")
    plt.legend()
    plt.tight_layout()
    p_hist = os.path.join(args.outdir, "class_hist_train_vs_test.png")
    plt.savefig(p_hist, dpi=200)
    plt.close()
    print("Saved:", p_hist)

    # --- Window-within-gesture-interval check (on RAW test subject) ---
    # We verify if each window lies entirely inside one constant restimulus interval.
    Xraw, yraw, rep = load_subject(args.test_subject, args.base)
    s_idx, e_idx, y_center = window_by_repetition_with_indices(
        Xraw, yraw, rep, win_ms=args.win_ms, overlap=args.overlap
    )

    # ignore rest (0) like your pipeline
    keep = (y_center != 0)
    s_idx, e_idx, y_center = s_idx[keep], e_idx[keep], y_center[keep]

    # check if y is constant inside each window
    ok = 0
    for s, e in zip(s_idx, e_idx):
        yy = yraw[s:e]
        if np.all(yy == yy[0]) and yy[0] != 0:
            ok += 1
    frac_ok = ok / len(s_idx) if len(s_idx) else 0.0

    print("\n=== Window boundary check (test subject raw) ===")
    print("Subject:", args.test_subject)
    print("Windows checked (non-rest):", int(len(s_idx)))
    print("Fraction fully inside a single gesture interval:", float(frac_ok))

    # save a small text summary
    p_txt = os.path.join(args.outdir, "window_interval_check.txt")
    with open(p_txt, "w", encoding="utf-8") as f:
        f.write(f"subject={args.test_subject}\n")
        f.write(f"windows_non_rest={int(len(s_idx))}\n")
        f.write(f"fraction_inside_single_gesture_interval={float(frac_ok)}\n")
    print("Saved:", p_txt)

if __name__ == "__main__":
    main()
>>>>>>> 9bbfa3b24bb49262e519b5672b0b6636e2cc4682
