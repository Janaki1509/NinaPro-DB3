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
    emg_list, y_list = [], []
    for fp in mats:
        m = loadmat(fp)
        emg_list.append(m["emg"].astype(np.float32))                  # (T,12)
        y_list.append(m["restimulus"].reshape(-1).astype(np.int32))   # (T,)
    X = np.vstack(emg_list)
    y = np.concatenate(y_list)
    return X, y

def find_first_interval(y, gesture_id: int, min_len: int):
    y = y.reshape(-1)
    idx = np.where(y == gesture_id)[0]
    if len(idx) == 0:
        return None
    # find contiguous run containing the first index
    i0 = idx[0]
    s = i0
    while s > 0 and y[s-1] == gesture_id:
        s -= 1
    e = i0
    while e < len(y) and y[e] == gesture_id:
        e += 1
    if (e - s) < min_len:
        # try to find another run
        runs = []
        i = 0
        while i < len(y):
            if y[i] == gesture_id:
                j = i
                while j < len(y) and y[j] == gesture_id:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        for s, e in runs:
            if (e - s) >= min_len:
                return (s, e)
        return None
    return (s, e)

def offset_plot(ax, Xwin, title):
    # Xwin: (T,12). plot 12 channels with vertical offsets
    T, C = Xwin.shape
    t = np.arange(T)
    # normalize per channel for display
    Xn = Xwin - Xwin.mean(axis=0, keepdims=True)
    Xn = Xn / (Xn.std(axis=0, keepdims=True) + 1e-8)
    offset = 3.0
    for c in range(C):
        ax.plot(t, Xn[:, c] + c*offset)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Channels (offset)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--subjA", required=True)
    ap.add_argument("--subjB", required=True)
    ap.add_argument("--gesture", type=int, required=True, help="gesture id in restimulus (non-zero)")
    ap.add_argument("--win_len", type=int, default=300, help="samples (150ms at 2000Hz)")
    ap.add_argument("--out", default=r"outputs\plots\raw_emg_two_subjects.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    XA, yA = load_subject(args.subjA, args.base)
    XB, yB = load_subject(args.subjB, args.base)

    ia = find_first_interval(yA, args.gesture, args.win_len)
    ib = find_first_interval(yB, args.gesture, args.win_len)
    if ia is None or ib is None:
        raise SystemExit("Could not find a long enough interval for that gesture in one of the subjects.")

    sa, ea = ia
    sb, eb = ib

    XAw = XA[sa:sa+args.win_len, :]   # (T,12)
    XBw = XB[sb:sb+args.win_len, :]

    fig, axes = plt.subplots(1, 2, figsize=(12,4), sharey=True)
    offset_plot(axes[0], XAw, f"{args.subjA} gesture={args.gesture}")
    offset_plot(axes[1], XBw, f"{args.subjB} gesture={args.gesture}")
    fig.suptitle("Raw EMG Windows Across Two Subjects (12 channels, offset)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print("Saved:", args.out)

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
    emg_list, y_list = [], []
    for fp in mats:
        m = loadmat(fp)
        emg_list.append(m["emg"].astype(np.float32))                  # (T,12)
        y_list.append(m["restimulus"].reshape(-1).astype(np.int32))   # (T,)
    X = np.vstack(emg_list)
    y = np.concatenate(y_list)
    return X, y

def find_first_interval(y, gesture_id: int, min_len: int):
    y = y.reshape(-1)
    idx = np.where(y == gesture_id)[0]
    if len(idx) == 0:
        return None
    # find contiguous run containing the first index
    i0 = idx[0]
    s = i0
    while s > 0 and y[s-1] == gesture_id:
        s -= 1
    e = i0
    while e < len(y) and y[e] == gesture_id:
        e += 1
    if (e - s) < min_len:
        # try to find another run
        runs = []
        i = 0
        while i < len(y):
            if y[i] == gesture_id:
                j = i
                while j < len(y) and y[j] == gesture_id:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        for s, e in runs:
            if (e - s) >= min_len:
                return (s, e)
        return None
    return (s, e)

def offset_plot(ax, Xwin, title):
    # Xwin: (T,12). plot 12 channels with vertical offsets
    T, C = Xwin.shape
    t = np.arange(T)
    # normalize per channel for display
    Xn = Xwin - Xwin.mean(axis=0, keepdims=True)
    Xn = Xn / (Xn.std(axis=0, keepdims=True) + 1e-8)
    offset = 3.0
    for c in range(C):
        ax.plot(t, Xn[:, c] + c*offset)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Channels (offset)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--subjA", required=True)
    ap.add_argument("--subjB", required=True)
    ap.add_argument("--gesture", type=int, required=True, help="gesture id in restimulus (non-zero)")
    ap.add_argument("--win_len", type=int, default=300, help="samples (150ms at 2000Hz)")
    ap.add_argument("--out", default=r"outputs\plots\raw_emg_two_subjects.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    XA, yA = load_subject(args.subjA, args.base)
    XB, yB = load_subject(args.subjB, args.base)

    ia = find_first_interval(yA, args.gesture, args.win_len)
    ib = find_first_interval(yB, args.gesture, args.win_len)
    if ia is None or ib is None:
        raise SystemExit("Could not find a long enough interval for that gesture in one of the subjects.")

    sa, ea = ia
    sb, eb = ib

    XAw = XA[sa:sa+args.win_len, :]   # (T,12)
    XBw = XB[sb:sb+args.win_len, :]

    fig, axes = plt.subplots(1, 2, figsize=(12,4), sharey=True)
    offset_plot(axes[0], XAw, f"{args.subjA} gesture={args.gesture}")
    offset_plot(axes[1], XBw, f"{args.subjB} gesture={args.gesture}")
    fig.suptitle("Raw EMG Windows Across Two Subjects (12 channels, offset)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
>>>>>>> 9bbfa3b24bb49262e519b5672b0b6636e2cc4682
