#!/usr/bin/env python3
import csv, math
import numpy as np

path = "results_loso_baselines.csv"
rows = []
with open(path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

def col(name):
    return np.array([float(x[name]) for x in rows], dtype=float)

maj = col("maj_acc")
pro = col("proto_acc")
sof = col("soft_acc")
std_ratio = col("std_ratio")
shift = np.abs(np.log(std_ratio))  # distance from 1.0 on log scale

print("N subjects:", len(rows))
print("\nMean ± std (accuracy):")
print("majority :", float(maj.mean()), "±", float(maj.std()))
print("prototype:", float(pro.mean()), "±", float(pro.std()))
print("softmax  :", float(sof.mean()), "±", float(sof.std()))

print("\nBest softmax subjects:")
idx = np.argsort(-sof)[:5]
for i in idx:
    print(rows[i]["test_subject"], "soft", sof[i], "std_ratio", std_ratio[i], "shift", shift[i])

print("\nWorst softmax subjects:")
idx = np.argsort(sof)[:5]
for i in idx:
    print(rows[i]["test_subject"], "soft", sof[i], "std_ratio", std_ratio[i], "shift", shift[i])

# correlation with domain shift
def corr(a,b):
    a=a-a.mean(); b=b-b.mean()
    return float((a*b).sum() / (math.sqrt((a*a).sum())*math.sqrt((b*b).sum()) + 1e-12))

print("\nCorrelation (shift vs accuracy):")
print("shift vs prototype:", corr(shift, pro))
print("shift vs softmax  :", corr(shift, sof))
