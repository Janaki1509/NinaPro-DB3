#!/usr/bin/env python3
import os, glob, json, csv, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/per_subject")
    ap.add_argument("--out_csv", default="results/loso_minirocket.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.json")))
    if not files:
        raise SystemExit(f"No JSON files found in {args.in_dir}")

    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    # stable column order
    cols = [
        "test_subject", "N_train", "N_test", "K",
        "win_ms", "overlap", "strict_purity",
        "train_per_class", "test_per_class", "num_kernels",
        "std_ratio", "acc", "balanced_acc", "macro_f1"
    ]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print("Merged", len(rows), "files ->", args.out_csv)

if __name__ == "__main__":
    main()