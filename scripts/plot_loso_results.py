<<<<<<< HEAD
#!/usr/bin/env python3
import os, csv, math
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "results_loso_baselines.csv"
OUTDIR = os.path.join("outputs", "plots")
os.makedirs(OUTDIR, exist_ok=True)

rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

subjects = [x["test_subject"] for x in rows]
maj = np.array([float(x["maj_acc"]) for x in rows])
pro = np.array([float(x["proto_acc"]) for x in rows])
sof = np.array([float(x["soft_acc"]) for x in rows])
std_ratio = np.array([float(x["std_ratio"]) for x in rows])

# --- Plot 1: accuracy by subject ---
x = np.arange(len(subjects))

plt.figure()
plt.plot(x, maj, marker="o", label="Majority")
plt.plot(x, pro, marker="o", label="Prototype (cosine)")
plt.plot(x, sof, marker="o", label="Handcrafted + Linear (softmax)")
plt.xticks(x, subjects, rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("LOSO Accuracy by Test Subject")
plt.legend()
plt.tight_layout()
p1 = os.path.join(OUTDIR, "loso_accuracy_by_subject.png")
plt.savefig(p1, dpi=200)
plt.close()

# --- Plot 2: domain shift vs accuracy ---
# distance from 1.0 in log-space (0 = no shift)
shift = np.abs(np.log(std_ratio + 1e-12))

plt.figure()
plt.scatter(shift, pro, label="Prototype (cosine)")
plt.scatter(shift, sof, label="Handcrafted + Linear (softmax)")
plt.xlabel("|log(std_ratio)|  (0 = no shift)")
plt.ylabel("Accuracy")
plt.title("Domain Shift vs LOSO Accuracy")
plt.legend()
plt.tight_layout()
p2 = os.path.join(OUTDIR, "domain_shift_vs_accuracy.png")
plt.savefig(p2, dpi=200)
plt.close()

print("Saved:", p1)
print("Saved:", p2)
=======
#!/usr/bin/env python3
import os, csv, math
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "results_loso_baselines.csv"
OUTDIR = os.path.join("outputs", "plots")
os.makedirs(OUTDIR, exist_ok=True)

rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

subjects = [x["test_subject"] for x in rows]
maj = np.array([float(x["maj_acc"]) for x in rows])
pro = np.array([float(x["proto_acc"]) for x in rows])
sof = np.array([float(x["soft_acc"]) for x in rows])
std_ratio = np.array([float(x["std_ratio"]) for x in rows])

# --- Plot 1: accuracy by subject ---
x = np.arange(len(subjects))

plt.figure()
plt.plot(x, maj, marker="o", label="Majority")
plt.plot(x, pro, marker="o", label="Prototype (cosine)")
plt.plot(x, sof, marker="o", label="Handcrafted + Linear (softmax)")
plt.xticks(x, subjects, rotation=45, ha="right")
plt.ylabel("Accuracy")
plt.title("LOSO Accuracy by Test Subject")
plt.legend()
plt.tight_layout()
p1 = os.path.join(OUTDIR, "loso_accuracy_by_subject.png")
plt.savefig(p1, dpi=200)
plt.close()

# --- Plot 2: domain shift vs accuracy ---
# distance from 1.0 in log-space (0 = no shift)
shift = np.abs(np.log(std_ratio + 1e-12))

plt.figure()
plt.scatter(shift, pro, label="Prototype (cosine)")
plt.scatter(shift, sof, label="Handcrafted + Linear (softmax)")
plt.xlabel("|log(std_ratio)|  (0 = no shift)")
plt.ylabel("Accuracy")
plt.title("Domain Shift vs LOSO Accuracy")
plt.legend()
plt.tight_layout()
p2 = os.path.join(OUTDIR, "domain_shift_vs_accuracy.png")
plt.savefig(p2, dpi=200)
plt.close()

print("Saved:", p1)
print("Saved:", p2)
>>>>>>> 9bbfa3b24bb49262e519b5672b0b6636e2cc4682
