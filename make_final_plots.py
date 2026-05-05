import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"C:\ninapro_db3"
RESULTS = os.path.join(BASE, "results")
OUTDIR = os.path.join(BASE, "outputs", "plots")
os.makedirs(OUTDIR, exist_ok=True)

baseline_path = os.path.join(BASE, "results_loso_baselines.csv")
minirocket_path = os.path.join(RESULTS, "loso_minirocket_full.csv")
mlp_path = os.path.join(RESULTS, "loso_mlp.csv")

baseline = pd.read_csv(baseline_path)
minirocket = pd.read_csv(minirocket_path)
mlp = pd.read_csv(mlp_path)

# -------------------------
# Plot 1: mean LOSO bAcc by method
# -------------------------
comparison_all = pd.DataFrame({
    "Method": [
        "Majority",
        "Prototype",
        "Handcrafted + Softmax",
        "MiniROCKET",
        "MLP SoftMax",
    ],
    "Mean_bAcc": [
        baseline["maj_bacc"].mean(),
        baseline["proto_bacc"].mean(),
        baseline["soft_bacc"].mean(),
        minirocket["balanced_acc"].mean(),
        mlp["balanced_acc"].mean(),
    ]
})

plt.figure(figsize=(8, 5))
plt.bar(comparison_all["Method"], comparison_all["Mean_bAcc"])
plt.ylabel("Mean LOSO Balanced Accuracy")
plt.title("Method Comparison on NinaPro DB3 LOSO")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "method_comparison_loso_bacc.png"), dpi=200)
plt.close()

# -------------------------
# Plot 2: per-subject MiniROCKET vs MLP
# -------------------------
mini_plot = minirocket[["test_subject", "balanced_acc"]].copy()
mini_plot = mini_plot.rename(columns={"balanced_acc": "MiniROCKET"})
mlp_plot = mlp[["test_subject", "balanced_acc"]].copy()
mlp_plot = mlp_plot.rename(columns={"balanced_acc": "MLP SoftMax"})

merged = pd.merge(mini_plot, mlp_plot, on="test_subject", how="inner")
merged["subject_num"] = merged["test_subject"].str.extract(r"(\d+)").astype(int)
merged = merged.sort_values("subject_num")

x = range(len(merged))
width = 0.4

plt.figure(figsize=(10, 5))
plt.bar([i - width/2 for i in x], merged["MiniROCKET"], width=width, label="MiniROCKET")
plt.bar([i + width/2 for i in x], merged["MLP SoftMax"], width=width, label="MLP SoftMax")
plt.xticks(list(x), merged["test_subject"])
plt.ylabel("Balanced Accuracy")
plt.title("Per-Subject LOSO Balanced Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "per_subject_minirocket_vs_mlp.png"), dpi=200)
plt.close()

# -------------------------
# Plot 3: std_ratio vs MLP balanced accuracy
# -------------------------
plt.figure(figsize=(7, 5))
plt.scatter(mlp["std_ratio"], mlp["balanced_acc"])
for _, row in mlp.iterrows():
    plt.annotate(row["test_subject"], (row["std_ratio"], row["balanced_acc"]), fontsize=8)
plt.xlabel("std_ratio")
plt.ylabel("Balanced Accuracy")
plt.title("Domain Shift vs MLP Performance")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "std_ratio_vs_mlp_bacc.png"), dpi=200)
plt.close()

print("Saved plots to:", OUTDIR)
print("\nMethod means:")
print(comparison_all.to_string(index=False))