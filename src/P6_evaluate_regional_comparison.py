#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P6_eval_5hmc2stage_same_split_region.py

Train/evaluate 5hmC2Stage under the SAME fixed window split used by the
Deep5hmC-inspired regional comparator, then aggregate site-level predictions
back to 1-kb windows for region-level comparison.

Purpose
-------
This script closes the comparison loop for Table 4:
- Deep5hmC-inspired comparator: trains directly on region windows
- 5hmC2Stage comparator: trains at site x cell_type level, but uses the same
  fixed train/val/test window split, and is finally evaluated on aggregated
  region-level signals on the held-out test windows.

Key design
----------
1) Build eligible windows from stage2_csv + truth_csv
2) Make ONE fixed split at the window level using split_seed
3) Map site rows to train/val/test by window membership
4) Train a site-level classifier (default: LightGBM; optional XGBoost)
5) Predict p_pred on held-out TEST rows only
6) Aggregate p_pred and truth_cont to region-level windows
7) Report pooled / macro Spearman / Pearson / RMSE / MAE

Notes
-----
- This script assumes stage2_csv already contains the Stage-1 score as one
  of the numeric features, i.e. it is the actual 5hmC2Stage input table.
- This is the fairest way to compare 5hmC2Stage against a fixed-split regional
  comparator without changing 5hmC2Stage into a different model family.
"""

import os
import json
import argparse
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None


# ---------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_numeric(x):
    return pd.to_numeric(x, errors="coerce")


def parse_seed_list(x: str) -> List[int]:
    if not x.strip():
        return [42]
    return [int(i.strip()) for i in x.split(",") if i.strip()]


def safe_auc(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_pred))


def safe_ap(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)
    if np.unique(y_true).size < 2:
        return np.nan
    return float(average_precision_score(y_true, y_pred))


def safe_spearman(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return np.nan
    if np.unique(y_true[m]).size < 2 or np.unique(y_pred[m]).size < 2:
        return np.nan
    return float(spearmanr(y_true[m], y_pred[m]).statistic)


def safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return np.nan
    if np.unique(y_true[m]).size < 2 or np.unique(y_pred[m]).size < 2:
        return np.nan
    return float(pearsonr(y_true[m], y_pred[m])[0])


def safe_mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    return float(np.mean((y_true[m] - y_pred[m]) ** 2))


def safe_rmse(y_true, y_pred):
    x = safe_mse(y_true, y_pred)
    return float(np.sqrt(x)) if np.isfinite(x) else np.nan


def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


# ---------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------

def load_stage2_table(stage2_csv: str, label_col: str, weight_col: str) -> pd.DataFrame:
    df = pd.read_csv(stage2_csv, low_memory=False)
    need = ["chrom", "pos_based", "cell_type", label_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"stage2_csv missing required columns: {miss}")

    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["pos_based"] = safe_numeric(df["pos_based"])
    df = df[df["pos_based"].notna()].copy()
    df["pos_based"] = df["pos_based"].astype(int)
    df["cell_type"] = df["cell_type"].astype(str)
    df[label_col] = safe_numeric(df[label_col]).astype(int)

    if weight_col and weight_col in df.columns:
        df["site_weight"] = safe_numeric(df[weight_col]).fillna(1.0).astype(float)
    else:
        df["site_weight"] = 1.0

    return df


def load_truth_long(truth_csv: str, truth_signal_col: str, weight_col: str) -> pd.DataFrame:
    df = pd.read_csv(truth_csv, low_memory=False)
    need = ["chrom", "pos_based", "cell_type", truth_signal_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"truth_csv missing required columns: {miss}")

    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["pos_based"] = safe_numeric(df["pos_based"])
    df = df[df["pos_based"].notna()].copy()
    df["pos_based"] = df["pos_based"].astype(int)
    df["cell_type"] = df["cell_type"].astype(str)
    df[truth_signal_col] = safe_numeric(df[truth_signal_col]).astype(float)

    if weight_col and weight_col in df.columns:
        df["row_weight"] = safe_numeric(df[weight_col]).fillna(1.0).astype(float)
    else:
        df["row_weight"] = 1.0

    return df[["chrom", "pos_based", "cell_type", truth_signal_col, "row_weight"]].copy()


# ---------------------------------------------------------------------
# feature selection and windows
# ---------------------------------------------------------------------

def pick_feature_cols(df: pd.DataFrame, label_col: str) -> List[str]:
    exclude = {
        "chrom", "pos_based", "cell_type",
        label_col, "label_site",
        "sample_weight", "site_weight",
        "gene_id", "gene_name",
    }
    feat_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feat_cols.append(c)
    if len(feat_cols) == 0:
        raise RuntimeError("No numeric feature columns found in stage2_csv after exclusion.")
    return feat_cols


def attach_windows(df: pd.DataFrame, window_bp: int) -> pd.DataFrame:
    out = df.copy()
    out["window_start"] = ((out["pos_based"] - 1) // int(window_bp)) * int(window_bp) + 1
    out["window_end"] = out["window_start"] + int(window_bp) - 1
    out["split_group"] = out["chrom"].astype(str) + "|" + out["window_start"].astype(str)
    return out


def build_merged_table(stage2_df: pd.DataFrame, truth_df: pd.DataFrame, truth_signal_col: str, window_bp: int) -> pd.DataFrame:
    df = stage2_df.merge(truth_df, on=["chrom", "pos_based", "cell_type"], how="inner")
    if len(df) == 0:
        raise RuntimeError("No merged rows between stage2_csv and truth_csv.")
    df = attach_windows(df, window_bp)
    return df


def build_eligible_windows(merged_df: pd.DataFrame, min_cpgs: int) -> pd.DataFrame:
    rows = []
    for keys, sub in merged_df.groupby(["chrom", "window_start", "window_end", "cell_type", "split_group"], sort=False):
        chrom, ws, we, ct, sg = keys
        n_cpgs = int(len(sub))
        if n_cpgs < int(min_cpgs):
            continue
        rows.append({
            "chrom": chrom,
            "window_start": int(ws),
            "window_end": int(we),
            "cell_type": ct,
            "split_group": sg,
            "n_cpgs": n_cpgs,
        })
    win_df = pd.DataFrame(rows)
    if len(win_df) == 0:
        raise RuntimeError("No eligible windows found after min_cpgs filtering.")
    return win_df


# ---------------------------------------------------------------------
# fixed split
# ---------------------------------------------------------------------

def make_fixed_window_split(win_df: pd.DataFrame, split_seed: int, test_size: float, val_size: float):
    idx = np.arange(len(win_df))
    groups = win_df["split_group"].astype(str).to_numpy()
    dummy_y = np.zeros(len(win_df), dtype=int)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
    trva_idx, te_idx = next(gss1.split(idx, dummy_y, groups))

    groups_trva = groups[trva_idx]
    val_frac_in_trva = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_in_trva, random_state=split_seed + 1)
    tr_rel, va_rel = next(gss2.split(trva_idx, dummy_y[trva_idx], groups_trva))
    tr_idx = trva_idx[tr_rel]
    va_idx = trva_idx[va_rel]

    split_map = {}
    for i in tr_idx:
        split_map[(win_df.iloc[i]["split_group"], win_df.iloc[i]["cell_type"])] = "train"
    for i in va_idx:
        split_map[(win_df.iloc[i]["split_group"], win_df.iloc[i]["cell_type"])] = "val"
    for i in te_idx:
        split_map[(win_df.iloc[i]["split_group"], win_df.iloc[i]["cell_type"])] = "test"

    split_win_df = win_df.copy()
    split_win_df["split"] = [split_map[(sg, ct)] for sg, ct in zip(split_win_df["split_group"], split_win_df["cell_type"])]
    return split_win_df


def attach_site_split(merged_df: pd.DataFrame, split_win_df: pd.DataFrame) -> pd.DataFrame:
    key_df = split_win_df[["split_group", "cell_type", "split"]].drop_duplicates()
    out = merged_df.merge(key_df, on=["split_group", "cell_type"], how="inner")
    if len(out) == 0:
        raise RuntimeError("No site rows remain after attaching split information.")
    return out


# ---------------------------------------------------------------------
# model training
# ---------------------------------------------------------------------

def fit_imputer(Xtr: pd.DataFrame):
    imp = SimpleImputer(strategy="median")
    Xtr2 = imp.fit_transform(Xtr)
    return imp, Xtr2


def transform_imputer(X: pd.DataFrame, imp):
    return imp.transform(X)


def train_one_seed(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   feat_cols: List[str], label_col: str, model_name: str, seed: int):
    Xtr = train_df[feat_cols].copy()
    Xva = val_df[feat_cols].copy()
    Xte = test_df[feat_cols].copy()

    ytr = train_df[label_col].astype(int).to_numpy()
    yva = val_df[label_col].astype(int).to_numpy()
    yte = test_df[label_col].astype(int).to_numpy()

    wtr = train_df["site_weight"].astype(float).to_numpy()
    wva = val_df["site_weight"].astype(float).to_numpy()

    imp, Xtr2 = fit_imputer(Xtr)
    Xva2 = transform_imputer(Xva, imp)
    Xte2 = transform_imputer(Xte, imp)

    if model_name == "lgb":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed, but --model lgb was requested.")
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_samples=50,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(
            Xtr2, ytr,
            sample_weight=wtr,
            eval_set=[(Xva2, yva)],
            eval_sample_weight=[wva],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        pva = clf.predict_proba(Xva2)[:, 1]
        pte = clf.predict_proba(Xte2)[:, 1]
        best_iter = getattr(clf, "best_iteration_", None)

    elif model_name == "xgb":
        if xgb is None:
            raise RuntimeError("xgboost is not installed, but --model xgb was requested.")
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=4000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="auc",
        )
        clf.fit(
            Xtr2, ytr,
            sample_weight=wtr,
            eval_set=[(Xva2, yva)],
            sample_weight_eval_set=[wva],
            verbose=False,
        )
        pva = clf.predict_proba(Xva2)[:, 1]
        pte = clf.predict_proba(Xte2)[:, 1]
        best_iter = getattr(clf, "best_iteration", None)
    else:
        raise ValueError("--model must be one of: lgb, xgb")

    out = {
        "seed": seed,
        "val_auc": safe_auc(yva, pva),
        "val_ap": safe_ap(yva, pva),
        "test_auc_site": safe_auc(yte, pte),
        "test_ap_site": safe_ap(yte, pte),
        "best_iter": best_iter,
    }

    pred_df = test_df[["chrom", "pos_based", "cell_type", "window_start", "window_end", "split_group", "row_weight", "hmc_signal_cont"]].copy()
    pred_df["p_pred"] = pte
    return out, pred_df


# ---------------------------------------------------------------------
# region aggregation + eval
# ---------------------------------------------------------------------

def aggregate_region_eval(pred_df: pd.DataFrame):
    rows = []
    gkeys = ["chrom", "window_start", "window_end", "cell_type"]

    for keys, sub in pred_df.groupby(gkeys, sort=False):
        chrom, ws, we, ct = keys
        w = safe_numeric(sub["row_weight"]).fillna(1.0).to_numpy(dtype=float)
        truth = safe_numeric(sub["hmc_signal_cont"]).to_numpy(dtype=float)
        pred = safe_numeric(sub["p_pred"]).to_numpy(dtype=float)

        mask_t = np.isfinite(truth) & np.isfinite(w) & (w > 0)
        mask_p = np.isfinite(pred) & np.isfinite(w) & (w > 0)
        if mask_t.sum() == 0 or mask_p.sum() == 0:
            continue

        truth_sum = float(np.sum(truth[mask_t] * w[mask_t]))
        pred_sum = float(np.sum(pred[mask_p] * w[mask_p]))

        rows.append({
            "chrom": chrom,
            "window_start": int(ws),
            "window_end": int(we),
            "cell_type": ct,
            "truth_window": np.log1p(max(truth_sum, 0.0)),
            "pred_window": np.log1p(max(pred_sum, 0.0)),
            "n_cpgs": int(len(sub)),
        })

    win_pred = pd.DataFrame(rows)
    if len(win_pred) == 0:
        raise RuntimeError("No region-level rows available after aggregation.")

    pooled = {
        "pooled_spearman": safe_spearman(win_pred["truth_window"], win_pred["pred_window"]),
        "pooled_pearson": safe_pearson(win_pred["truth_window"], win_pred["pred_window"]),
        "pooled_rmse": safe_rmse(win_pred["truth_window"], win_pred["pred_window"]),
        "pooled_mae": safe_mae(win_pred["truth_window"], win_pred["pred_window"]),
        "n_windows": int(len(win_pred)),
    }

    per_group_rows = []
    for ct, sub in win_pred.groupby("cell_type", sort=False):
        per_group_rows.append({
            "cell_type": ct,
            "n_windows": int(len(sub)),
            "spearman": safe_spearman(sub["truth_window"], sub["pred_window"]),
            "pearson": safe_pearson(sub["truth_window"], sub["pred_window"]),
            "rmse": safe_rmse(sub["truth_window"], sub["pred_window"]),
            "mae": safe_mae(sub["truth_window"], sub["pred_window"]),
        })
    per_group_df = pd.DataFrame(per_group_rows)

    pooled.update({
        "macro_mean_spearman": float(per_group_df["spearman"].mean()),
        "macro_median_spearman": float(per_group_df["spearman"].median()),
        "macro_mean_pearson": float(per_group_df["pearson"].mean()),
        "macro_median_pearson": float(per_group_df["pearson"].median()),
        "macro_mean_rmse": float(per_group_df["rmse"].mean()),
        "macro_mean_mae": float(per_group_df["mae"].mean()),
        "n_groups": int(per_group_df["cell_type"].nunique()),
    })
    return pooled, per_group_df, win_pred


def summarize(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    row = {"n_runs": int(len(df))}
    for c in metric_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        row[f"{c}_mean"] = float(vals.mean()) if len(vals.dropna()) > 0 else np.nan
        row[f"{c}_std"] = float(vals.std(ddof=1)) if len(vals.dropna()) >= 2 else np.nan
        row[f"{c}_min"] = float(vals.min()) if len(vals.dropna()) > 0 else np.nan
        row[f"{c}_max"] = float(vals.max()) if len(vals.dropna()) > 0 else np.nan
    return pd.DataFrame([row])


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    import argparse
    import os
    import json
    import pandas as pd
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_csv", required=True)
    ap.add_argument("--truth_csv", required=True)
    ap.add_argument("--truth_signal_col", default="hmc_signal_cont")
    ap.add_argument("--label_col", default="label_cell")
    ap.add_argument("--weight_col", default="sample_weight")
    ap.add_argument("--window_bp", type=int, default=1000)
    ap.add_argument("--min_cpgs", type=int, default=3)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--seed_list", default="42,52,62,72,82")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--model", default="lgb", choices=["lgb", "xgb"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    seeds = parse_seed_list(args.seed_list)

    stage2_df = load_stage2_table(args.stage2_csv, args.label_col, args.weight_col)
    truth_df = load_truth_long(args.truth_csv, args.truth_signal_col, args.weight_col)
    merged_df = build_merged_table(stage2_df, truth_df, args.truth_signal_col, args.window_bp)
    win_df = build_eligible_windows(merged_df, args.min_cpgs)
    split_win_df = make_fixed_window_split(win_df, args.split_seed, args.test_size, args.val_size)
    split_site_df = attach_site_split(merged_df, split_win_df)
    feat_cols = pick_feature_cols(split_site_df, args.label_col)

    # save split membership for auditability
    split_win_df.to_csv(os.path.join(args.out_dir, "fixed_split_windows.csv"), index=False)
    split_site_df[["chrom", "pos_based", "cell_type", "window_start", "window_end", "split_group", "split"]].to_csv(
        os.path.join(args.out_dir, "fixed_split_site_rows.csv"), index=False
    )

    print("=" * 100)
    print("5hmC2Stage same-split regional evaluation")
    print("=" * 100)
    print(f"stage2_csv      : {args.stage2_csv}")
    print(f"truth_csv       : {args.truth_csv}")
    print(f"label_col       : {args.label_col}")
    print(f"window_bp       : {args.window_bp}")
    print(f"min_cpgs        : {args.min_cpgs}")
    print(f"split_seed      : {args.split_seed}")
    print(f"model           : {args.model}")
    print(f"eligible_windows: {len(split_win_df):,}")
    print(f"site_rows       : {len(split_site_df):,}")
    print(f"feature_cols    : {len(feat_cols)}")
    print(f"cell_types      : {sorted(split_site_df['cell_type'].unique().tolist())}")
    print("=" * 100)
    print("split counts (windows):")
    print(split_win_df["split"].value_counts().to_dict())
    print("split counts (site rows):")
    print(split_site_df["split"].value_counts().to_dict())

    train_df = split_site_df[split_site_df["split"] == "train"].copy()
    val_df = split_site_df[split_site_df["split"] == "val"].copy()
    test_df = split_site_df[split_site_df["split"] == "test"].copy()

    all_rows = []
    all_group_rows = []
    # 新增：用于收集所有seed的预测结果
    all_window_rows = []
    all_site_pred_rows = []

    for seed in seeds:
        print(f"\n===== Model seed {seed} =====")
        site_out, pred_df = train_one_seed(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feat_cols=feat_cols,
            label_col=args.label_col,
            model_name=args.model,
            seed=seed,
        )

        pooled, per_group_df, win_pred_df = aggregate_region_eval(pred_df)

        row = {
            "seed": seed,
            **site_out,
            **pooled,
        }
        all_rows.append(row)

        tmp = per_group_df.copy()
        tmp["seed"] = seed
        all_group_rows.append(tmp)

        # ========== 新增代码段：标记seed、保存单文件、存入全局列表 ==========
        pred_df["seed"] = seed
        win_pred_df["seed"] = seed

        # 单seed独立csv（保留原有逻辑，文件名同步对齐规范）
        pred_df.to_csv(
            os.path.join(args.out_dir, f"test_site_predictions_seed{seed}.csv"),
            index=False
        )
        win_pred_df.to_csv(
            os.path.join(args.out_dir, f"test_region_predictions_seed{seed}.csv"),
            index=False
        )

        # 追加至全局列表，循环结束后合并
        all_site_pred_rows.append(pred_df)
        all_window_rows.append(win_pred_df)
        # =================================================================

        print(
            f"site_test_auc={site_out['test_auc_site']:.4f} | "
            f"site_test_ap={site_out['test_ap_site']:.4f} | "
            f"region_pooled_s={pooled['pooled_spearman']:.4f} | "
            f"region_macro_mean_s={pooled['macro_mean_spearman']:.4f}"
        )

    # ========== 循环结束后：合并所有seed预测结果并保存 ==========
    if len(all_site_pred_rows) > 0:
        all_site = pd.concat(all_site_pred_rows, axis=0, ignore_index=True)
        all_site.to_csv(os.path.join(args.out_dir, "site_predictions_by_seed.csv"), index=False)

    if len(all_window_rows) > 0:
        all_win = pd.concat(all_window_rows, axis=0, ignore_index=True)
        all_win.to_csv(os.path.join(args.out_dir, "window_predictions_by_seed.csv"), index=False)

        # 按窗口分组，计算多seed预测均值
        mean_win = (
            all_win.groupby(["chrom", "window_start", "window_end", "cell_type"], as_index=False)
            .agg(
                truth_window=("truth_window", "first"),
                pred_window=("pred_window", "mean"),
                n_cpgs=("n_cpgs", "first"),
                n_runs=("pred_window", "count"),
            )
        )
        mean_win.to_csv(os.path.join(args.out_dir, "window_predictions_mean.csv"), index=False)
    # ===========================================================

    by_seed = pd.DataFrame(all_rows)
    by_seed.to_csv(os.path.join(args.out_dir, "summary_by_seed.csv"), index=False)

    metric_cols = [
        "val_auc", "val_ap", "test_auc_site", "test_ap_site",
        "pooled_spearman", "pooled_pearson", "pooled_rmse", "pooled_mae",
        "macro_mean_spearman", "macro_median_spearman",
        "macro_mean_pearson", "macro_median_pearson",
        "macro_mean_rmse", "macro_mean_mae",
    ]
    agg = summarize(by_seed, metric_cols)
    agg.to_csv(os.path.join(args.out_dir, "summary_agg.csv"), index=False)

    if len(all_group_rows) > 0:
        per_group = pd.concat(all_group_rows, axis=0, ignore_index=True)
        per_group.to_csv(os.path.join(args.out_dir, "summary_per_group_by_seed.csv"), index=False)

        rows = []
        for grp, sub in per_group.groupby("cell_type", sort=False):
            rows.append({
                "cell_type": grp,
                "n_runs": int(len(sub)),
                "spearman_mean": float(pd.to_numeric(sub["spearman"], errors="coerce").mean()),
                "spearman_std": float(pd.to_numeric(sub["spearman"], errors="coerce").std(ddof=1)) if len(
                    sub) >= 2 else np.nan,
                "pearson_mean": float(pd.to_numeric(sub["pearson"], errors="coerce").mean()),
                "pearson_std": float(pd.to_numeric(sub["pearson"], errors="coerce").std(ddof=1)) if len(
                    sub) >= 2 else np.nan,
                "rmse_mean": float(pd.to_numeric(sub["rmse"], errors="coerce").mean()),
                "rmse_std": float(pd.to_numeric(sub["rmse"], errors="coerce").std(ddof=1)) if len(sub) >= 2 else np.nan,
                "mae_mean": float(pd.to_numeric(sub["mae"], errors="coerce").mean()),
                "mae_std": float(pd.to_numeric(sub["mae"], errors="coerce").std(ddof=1)) if len(sub) >= 2 else np.nan,
            })
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "summary_per_group_agg.csv"), index=False)

    with open(os.path.join(args.out_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("\n✅ Finished.")
    print("Saved:")
    print(" -", os.path.join(args.out_dir, "fixed_split_windows.csv"))
    print(" -", os.path.join(args.out_dir, "fixed_split_site_rows.csv"))
    print(" -", os.path.join(args.out_dir, "summary_by_seed.csv"))
    print(" -", os.path.join(args.out_dir, "summary_agg.csv"))
    print(" -", os.path.join(args.out_dir, "summary_per_group_by_seed.csv"))
    print(" -", os.path.join(args.out_dir, "summary_per_group_agg.csv"))
    # 新增输出提示
    print(" -", os.path.join(args.out_dir, "site_predictions_by_seed.csv"))
    print(" -", os.path.join(args.out_dir, "window_predictions_by_seed.csv"))
    print(" -", os.path.join(args.out_dir, "window_predictions_mean.csv"))

if __name__ == "__main__":
    main()
