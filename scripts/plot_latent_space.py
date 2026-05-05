"""
plot_latent_space.py
====================
t-SNE visualization of shared latent space.
Run LOCALLY after: scp chandraj@port:~/ninapro_db3/runs/latent/*.npz ./runs/latent/

Produces 3 poster-ready figures:
  1. tsne_by_subject.png   — color=subject  (shows domain shift)
  2. tsne_by_gesture.png   — color=gesture  (shows class separability)
  3. tsne_panels.png       — one panel per gesture, each dot = subject (key figure)

Usage:
  python plot_latent_space.py --latent_dir ./runs/latent --out_dir ./outputs/plots
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


def load_all(latent_dir, n_subjects=11):
    latents, gesture_ids, subjects = [], [], []
    for s in range(1, n_subjects + 1):
        fp = os.path.join(latent_dir, f"s{s}_latents.npz")
        if not os.path.isfile(fp):
            print(f"  WARNING: missing {fp}")
            continue
        d = np.load(fp)
        latents.append(d["latents"])
        gesture_ids.append(d["gesture_ids"])   # original NinaPro gesture IDs
        subjects.append(np.full(len(d["latents"]), s, dtype=np.int32))
    if not latents:
        raise RuntimeError(f"No .npz files found in {latent_dir}")
    return (np.concatenate(latents),
            np.concatenate(gesture_ids),
            np.concatenate(subjects))


def subsample(latents, gesture_ids, subjects, max_n=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(latents)
    if n <= max_n:
        return latents, gesture_ids, subjects
    idx = rng.choice(n, max_n, replace=False)
    return latents[idx], gesture_ids[idx], subjects[idx]


def run_tsne(latents, seed=42):
    print("  Running t-SNE...", flush=True)
    return TSNE(n_components=2, random_state=seed, perplexity=40,
                n_iter=1000, learning_rate="auto", init="pca").fit_transform(latents)


def plot_by_subject(emb, subjects, out_path):
    unique = np.unique(subjects)
    cmap = cm.get_cmap("tab10", len(unique))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, s in enumerate(unique):
        m = subjects == s
        ax.scatter(emb[m, 0], emb[m, 1], c=[cmap(i)], s=5,
                   alpha=0.5, linewidths=0, label=f"S{s}")
    ax.set_title("Latent space colored by subject\n"
                 "(well-mixed = subject-invariant; clustered = domain shift)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(ncol=4, fontsize=8, loc="lower right", framealpha=0.8,
              markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_by_gesture(emb, gesture_ids, out_path):
    unique = np.unique(gesture_ids)
    cmap = cm.get_cmap("tab20", len(unique))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, g in enumerate(unique):
        m = gesture_ids == g
        ax.scatter(emb[m, 0], emb[m, 1], c=[cmap(i)], s=5,
                   alpha=0.5, linewidths=0)
    ax.set_title("Latent space colored by gesture class\n"
                 "(tight clusters = gesture-discriminative features)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_panels(emb, gesture_ids, subjects, out_path, top_n=6):
    """One panel per gesture — each dot colored by subject. KEY POSTER FIGURE."""
    counts = [(g, (gesture_ids == g).sum()) for g in np.unique(gesture_ids)]
    counts.sort(key=lambda x: -x[1])
    top_gestures = [g for g, _ in counts[:top_n]]
    unique_subjects = np.unique(subjects)
    cmap = cm.get_cmap("tab10", len(unique_subjects))

    ncols = 3
    nrows = (top_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    for pi, g in enumerate(top_gestures):
        ax = axes[pi]
        for i, s in enumerate(unique_subjects):
            m = (gesture_ids == g) & (subjects == s)
            if m.sum() == 0:
                continue
            ax.scatter(emb[m, 0], emb[m, 1], c=[cmap(i)], s=10,
                       alpha=0.75, linewidths=0, label=f"S{s}")
        ax.set_title(f"Gesture {g}", fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if pi == 0:
            ax.legend(ncol=3, fontsize=6, loc="best", framealpha=0.8,
                      markerscale=1.5)

    for ax in axes[top_n:]:
        ax.axis("off")

    fig.suptitle(
        "Per-gesture latent clusters — each color = subject\n"
        "Tighter within-gesture clustering across subjects = better domain alignment",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dir", default="./runs/latent")
    ap.add_argument("--out_dir",    default="./outputs/plots")
    ap.add_argument("--n_subjects", type=int, default=11)
    ap.add_argument("--max_samples", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading latent vectors...")
    latents, gesture_ids, subjects = load_all(args.latent_dir, args.n_subjects)
    print(f"  Total: {len(latents)} samples | "
          f"Subjects: {np.unique(subjects).tolist()} | "
          f"Gestures: {len(np.unique(gesture_ids))}")

    latents, gesture_ids, subjects = subsample(latents, gesture_ids, subjects, args.max_samples)
    print(f"  Subsampled to: {len(latents)}")

    emb = run_tsne(latents)

    plot_by_subject(emb, subjects,
                    os.path.join(args.out_dir, "tsne_by_subject.png"))
    plot_by_gesture(emb, gesture_ids,
                    os.path.join(args.out_dir, "tsne_by_gesture.png"))
    plot_panels(emb, gesture_ids, subjects,
                os.path.join(args.out_dir, "tsne_gesture_panels.png"))

    print("\nDone — use these 3 plots in your poster.")


if __name__ == "__main__":
    main()
