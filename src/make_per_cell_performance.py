import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

def safe_ap(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)

def collect(run_dir, model="lgb", label="baseline"):
    files = sorted(glob.glob(os.path.join(run_dir, "seed_*", model, "test_predictions.csv")))
    if not files:
        raise FileNotFoundError(f"No test_predictions.csv found under {run_dir}")

    rows = []
    for f in files:
        seed = os.path.basename(os.path.dirname(os.path.dirname(f))).replace("seed_", "")
        df = pd.read_csv(f)
        for ct, g in df.groupby("cell_type"):
            rows.append({
                "seed": int(seed),
                "cell_type": ct,
                f"{label}_auc": safe_auc(g["y_true"], g["p_pred"]),
                f"{label}_ap": safe_ap(g["y_true"], g["p_pred"]),
                f"{label}_n": len(g),
                f"{label}_pos": int(g["y_true"].sum())
            })
    return pd.DataFrame(rows)

def fmt_mean_sd(m, s):
    return f"{m:.4f} ± {s:.4f}"

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--baseline_dir", required=True)
ap.add_argument("--full_dir", required=True)
ap.add_argument("--out_csv", required=True)
ap.add_argument("--model", default="lgb")
args = ap.parse_args()

base = collect(args.baseline_dir, args.model, "baseline")
full = collect(args.full_dir, args.model, "full")

merged = base.merge(full, on=["seed", "cell_type"], how="inner")

summary = (
    merged.groupby("cell_type")
    .agg(
        baseline_auc_mean=("baseline_auc", "mean"),
        baseline_auc_sd=("baseline_auc", "std"),
        baseline_ap_mean=("baseline_ap", "mean"),
        baseline_ap_sd=("baseline_ap", "std"),
        full_auc_mean=("full_auc", "mean"),
        full_auc_sd=("full_auc", "std"),
        full_ap_mean=("full_ap", "mean"),
        full_ap_sd=("full_ap", "std"),
        mean_test_rows=("baseline_n", "mean"),
        mean_positive_rows=("baseline_pos", "mean"),
    )
    .reset_index()
)

out = pd.DataFrame({
    "Dataset": args.dataset,
    "Cell group": summary["cell_type"],
    "Baseline AUC": [fmt_mean_sd(m, s) for m, s in zip(summary["baseline_auc_mean"], summary["baseline_auc_sd"])],
    "Baseline AP": [fmt_mean_sd(m, s) for m, s in zip(summary["baseline_ap_mean"], summary["baseline_ap_sd"])],
    "Full 5hmC2Stage AUC": [fmt_mean_sd(m, s) for m, s in zip(summary["full_auc_mean"], summary["full_auc_sd"])],
    "Full 5hmC2Stage AP": [fmt_mean_sd(m, s) for m, s in zip(summary["full_ap_mean"], summary["full_ap_sd"])],
    "Mean test rows": summary["mean_test_rows"].round(1),
    "Mean positives": summary["mean_positive_rows"].round(1),
})

os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
out.to_csv(args.out_csv, index=False)
print(out.to_string(index=False))