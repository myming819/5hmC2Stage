#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P5_train_direction_style_baseline_enhanced.py

Enhanced DIRECTION-style one-stage baselines for 5hmC2Stage manuscript experiments.

Adds:
1. Multi-seed training in one run.
2. Grouped genomic block splitting with split-balance search.
3. RF / RBF-SVM / LightGBM one-stage baselines.
4. Strict leakage-control column filtering by default.
5. SVM uses median imputation + StandardScaler + RBF-SVC.
6. Rich outputs for NAR-ready figures and tables.

Recommended use
---------------
python P5_train_direction_style_baseline_enhanced.py \
  --task cell \
  --dataset_name Brain \
  --in_csv model_input/Brain_CPG_ALL_UNION/stage2_cell_static_dynamic_union.csv \
  --out_dir model_output/direction_style_enhanced/Brain \
  --models rf,lgb,svm \
  --seeds 42,43,44,45,46 \
  --block_bp 10000 \
  --use_sample_weight \
  --max_train_rows_svm 60000
"""

from __future__ import annotations

import os
import re
import json
import math
import argparse
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
)

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_seeds(seed_text: str) -> List[int]:
    vals = []
    for x in str(seed_text).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    if not vals:
        raise ValueError("--seeds cannot be empty")
    return vals


def to_jsonable(x):
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


def downcast_df_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="float")
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="integer")
    return out


def sanitize_feature_names(columns: Sequence[str]) -> List[str]:
    new_cols: List[str] = []
    seen: Dict[str, int] = {}
    for i, c in enumerate(columns):
        s = str(c)
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        if s == "":
            s = f"f_{i}"
        if s[0].isdigit():
            s = "f_" + s
        if s in seen:
            seen[s] += 1
            s = f"{s}__{seen[s]}"
        else:
            seen[s] = 0
        new_cols.append(s)
    return new_cols


def apply_safe_feature_names(Xtr: pd.DataFrame, Xva: pd.DataFrame, Xte: pd.DataFrame):
    orig_cols = list(Xtr.columns)
    safe_cols = sanitize_feature_names(orig_cols)
    rename_map = dict(zip(orig_cols, safe_cols))
    Xtr2 = Xtr.rename(columns=rename_map)
    Xva2 = Xva.rename(columns=rename_map)
    Xte2 = Xte.rename(columns=rename_map)
    feat_map = pd.DataFrame({"original_feature": orig_cols, "safe_feature": safe_cols})
    return Xtr2, Xva2, Xte2, feat_map


def make_groups(df: pd.DataFrame, block_bp: int = 10000) -> np.ndarray:
    if "chrom" not in df.columns or "pos_based" not in df.columns:
        raise ValueError("input must contain chrom and pos_based")
    chrom = df["chrom"].astype(str).str.strip()
    pos = pd.to_numeric(df["pos_based"], errors="coerce").fillna(-1).astype(np.int64)
    blk = ((pos - 1) // int(block_bp)).astype(np.int64)
    return (chrom + "|" + blk.astype(str)).astype(str).values


def _coerce_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return downcast_df_numeric(out)


BASE_DROP_COLS = {
    "chrom", "pos_based", "start", "end", "strand",
    "gene_id", "gene_name", "gene_id_base", "transcript_id",
    "site_id", "cpg_id", "probe_id", "feature_id",
    "label_site", "label_cell", "label", "target", "y", "y_true",
    "sample_weight", "label_confidence", "label_cell_mode",
    "site_role", "split", "fold", "row_id",
    "pos_count", "obs_count", "neg_count", "pos_ratio", "site_soft_score",
    "5hmc_signal", "hmC_signal", "hmc_signal",
    "rank_pct_cell", "zscore_cell", "thr_cell_used",
    "p_pred", "score", "prob", "prediction",
}

LEAKAGE_KEYWORDS = [
    "stage1", "propensity", "prior", "p_site", "u_site",
    "label_", "true_", "pred_", "prediction_", "target",
    "5hmc_signal", "hmc_signal", "rank_pct", "zscore", "thr_cell",
]


def find_auto_drop_columns(columns: Sequence[str], allow_stage1_features: bool = False) -> List[str]:
    out = []
    for c in columns:
        cl = str(c).lower()
        if c in BASE_DROP_COLS:
            out.append(c)
            continue
        if allow_stage1_features:
            blocked = [k for k in LEAKAGE_KEYWORDS if k not in {"stage1", "propensity", "prior", "p_site", "u_site"}]
        else:
            blocked = LEAKAGE_KEYWORDS
        if any(k in cl for k in blocked):
            if cl == "cell_type":
                continue
            out.append(c)
    return sorted(set(out))


def prepare_site_task(df: pd.DataFrame, args):
    if "label_site" not in df.columns:
        raise ValueError("site task requires column: label_site")
    y = pd.to_numeric(df["label_site"], errors="coerce")
    keep = y.notna()
    df = df.loc[keep].copy().reset_index(drop=True)
    y = y.loc[keep].astype(int).to_numpy()
    auto_drop = find_auto_drop_columns(df.columns, allow_stage1_features=args.allow_stage1_features)
    X = df.drop(columns=auto_drop, errors="ignore").copy()
    X = _coerce_numeric_frame(X)
    feature_cols = list(X.columns)
    meta_cols = [c for c in ["chrom", "pos_based"] if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)
    groups = make_groups(df, block_bp=args.block_bp)
    sample_weight = np.ones(len(df), dtype=float)
    return df, X, y, feature_cols, groups, meta, sample_weight, auto_drop


def prepare_cell_task(df: pd.DataFrame, args):
    if "label_cell" not in df.columns:
        raise ValueError("cell task requires column: label_cell")
    y = pd.to_numeric(df["label_cell"], errors="coerce")
    keep = y.notna()
    df = df.loc[keep].copy().reset_index(drop=True)
    y = y.loc[keep].astype(int).to_numpy()
    if "sample_weight" in df.columns:
        w = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        w = np.where(np.isfinite(w), w, 1.0)
        w = np.clip(w, 1e-6, None)
    else:
        w = np.ones(len(df), dtype=float)
    auto_drop = find_auto_drop_columns(df.columns, allow_stage1_features=args.allow_stage1_features)
    auto_drop = [c for c in auto_drop if c != "cell_type"]
    X = df.drop(columns=auto_drop, errors="ignore").copy()
    if args.no_cell_type_onehot:
        X = X.drop(columns=["cell_type"], errors="ignore")
    elif "cell_type" in X.columns:
        ct = pd.get_dummies(X["cell_type"].astype(str), prefix="ct", dtype=np.uint8)
        X = pd.concat([X.drop(columns=["cell_type"], errors="ignore"), ct], axis=1)
    X = _coerce_numeric_frame(X)
    feature_cols = list(X.columns)
    meta_cols = [c for c in ["chrom", "pos_based", "cell_type"] if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True)
    groups = make_groups(df, block_bp=args.block_bp)
    return df, X, y, feature_cols, groups, meta, w, auto_drop


def _split_diagnostics(indices: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str) -> Dict[str, float]:
    yy = y[indices]
    gg = groups[indices]
    return {
        f"{name}_n": int(len(indices)),
        f"{name}_pos": int(yy.sum()),
        f"{name}_neg": int(len(yy) - yy.sum()),
        f"{name}_pos_rate": float(np.mean(yy)) if len(yy) else float("nan"),
        f"{name}_n_blocks": int(len(pd.unique(gg))),
    }


def _overlap_count(groups: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
    return len(set(groups[a]).intersection(set(groups[b])))


def split_train_val_test_balanced(y, groups, seed, test_size, val_size, search_iter=200):
    n = len(y)
    idx = np.arange(n)
    overall_rate = float(np.mean(y))
    target_test_n = test_size * n
    target_val_n = val_size * n
    best = None
    best_score = float("inf")
    rng = np.random.RandomState(seed)
    for _ in range(max(1, int(search_iter))):
        rs1 = int(rng.randint(0, 2**31 - 1))
        rs2 = int(rng.randint(0, 2**31 - 1))
        try:
            gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs1)
            trainval_idx, test_idx = next(gss1.split(idx, y, groups))
            val_frac_in_trainval = val_size / max(1e-9, (1.0 - test_size))
            gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_in_trainval, random_state=rs2)
            tr_rel, va_rel = next(gss2.split(trainval_idx, y[trainval_idx], groups[trainval_idx]))
            train_idx = trainval_idx[tr_rel]
            val_idx = trainval_idx[va_rel]
        except Exception:
            continue
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        tr_rate = float(np.mean(y[train_idx]))
        va_rate = float(np.mean(y[val_idx]))
        te_rate = float(np.mean(y[test_idx]))
        score = (
            3.0 * abs(te_rate - overall_rate)
            + 2.0 * abs(va_rate - overall_rate)
            + 1.0 * abs(tr_rate - overall_rate)
            + 0.5 * abs(len(test_idx) - target_test_n) / max(1.0, target_test_n)
            + 0.5 * abs(len(val_idx) - target_val_n) / max(1.0, target_val_n)
            + 100.0 * (_overlap_count(groups, train_idx, val_idx)
                       + _overlap_count(groups, train_idx, test_idx)
                       + _overlap_count(groups, val_idx, test_idx))
        )
        if score < best_score:
            best_score = score
            best = (np.sort(train_idx), np.sort(val_idx), np.sort(test_idx))
    if best is None:
        raise RuntimeError("Could not find a valid grouped split containing both classes in every split.")
    train_idx, val_idx, test_idx = best
    diag = {
        "seed": int(seed),
        "split_search_score": float(best_score),
        "overall_n": int(n),
        "overall_pos": int(y.sum()),
        "overall_neg": int(n - y.sum()),
        "overall_pos_rate": overall_rate,
        "overall_n_blocks": int(len(pd.unique(groups))),
        **_split_diagnostics(train_idx, y, groups, "train"),
        **_split_diagnostics(val_idx, y, groups, "val"),
        **_split_diagnostics(test_idx, y, groups, "test"),
        "train_val_block_overlap": _overlap_count(groups, train_idx, val_idx),
        "train_test_block_overlap": _overlap_count(groups, train_idx, test_idx),
        "val_test_block_overlap": _overlap_count(groups, val_idx, test_idx),
    }
    return train_idx, val_idx, test_idx, diag


def maybe_subsample_train(train_idx, max_train_rows, seed, y=None):
    if max_train_rows <= 0 or len(train_idx) <= max_train_rows:
        return train_idx
    rng = np.random.RandomState(seed + 12345)
    if y is None:
        return np.sort(rng.choice(train_idx, size=max_train_rows, replace=False))
    yy = y[train_idx]
    pos_idx = train_idx[yy == 1]
    neg_idx = train_idx[yy == 0]
    pos_rate = len(pos_idx) / max(1, len(train_idx))
    n_pos = min(len(pos_idx), max(1, int(round(max_train_rows * pos_rate))))
    n_neg = min(len(neg_idx), max_train_rows - n_pos)
    chosen = []
    if n_pos > 0:
        chosen.append(rng.choice(pos_idx, size=n_pos, replace=False))
    if n_neg > 0:
        chosen.append(rng.choice(neg_idx, size=n_neg, replace=False))
    out = np.concatenate(chosen) if chosen else rng.choice(train_idx, size=max_train_rows, replace=False)
    return np.sort(out)


def align_frames(train_df, val_df, test_df):
    common = list(train_df.columns)
    return train_df, val_df.reindex(columns=common), test_df.reindex(columns=common)


def fit_rf(Xtr, ytr, seed, args, sample_weight=None):
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_depth=None if args.rf_max_depth <= 0 else args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            min_samples_split=args.rf_min_samples_split,
            max_features=args.rf_max_features,
            class_weight="balanced",
            random_state=seed,
            n_jobs=args.n_jobs,
        )),
    ])
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight
    clf.fit(Xtr, ytr, **fit_kwargs)
    return clf


def fit_svm(Xtr, ytr, seed, args, sample_weight=None):
    clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=args.svm_c,
            gamma=args.svm_gamma,
            probability=True,
            class_weight="balanced",
            random_state=seed,
            cache_size=args.svm_cache_size,
        )),
    ])
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight
    clf.fit(Xtr, ytr, **fit_kwargs)
    return clf


def fit_lgb(Xtr, ytr, Xva, yva, seed, args, sample_weight_tr=None, sample_weight_va=None):
    if lgb is None:
        raise RuntimeError("lightgbm is not installed. Install lightgbm or remove lgb from --models.")
    pos = max(1.0, float(np.sum(ytr == 1)))
    neg = max(1.0, float(np.sum(ytr == 0)))
    spw = neg / pos
    dtr = lgb.Dataset(Xtr, label=ytr, weight=sample_weight_tr, free_raw_data=True)
    dva = lgb.Dataset(Xva, label=yva, weight=sample_weight_va, reference=dtr, free_raw_data=True)
    params = {
        "objective": "binary",
        "metric": ["auc", "average_precision", "binary_logloss"],
        "learning_rate": args.lgb_lr,
        "num_leaves": args.lgb_num_leaves,
        "max_depth": args.lgb_max_depth,
        "min_data_in_leaf": args.lgb_min_data_in_leaf,
        "feature_fraction": args.lgb_feature_fraction,
        "bagging_fraction": args.lgb_bagging_fraction,
        "bagging_freq": args.lgb_bagging_freq,
        "lambda_l1": args.lgb_lambda_l1,
        "lambda_l2": args.lgb_lambda_l2,
        "scale_pos_weight": spw,
        "seed": seed,
        "data_random_seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "num_threads": args.n_jobs,
        "verbosity": -1,
        "force_col_wise": True,
    }
    return lgb.train(
        params,
        dtr,
        num_boost_round=args.lgb_estimators,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.lgb_early_stop, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def predict_proba(model_name, model, X):
    if model_name == "lgb":
        return np.asarray(model.predict(X, num_iteration=getattr(model, "best_iteration", None)), dtype=float)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def save_model(model_name, model, out_path):
    if model_name == "lgb":
        model.save_model(out_path)
    else:
        import joblib
        joblib.dump(model, out_path)


def save_feature_importance(model_name, model, feature_cols, out_csv):
    try:
        if model_name == "rf":
            imp = model.named_steps["clf"].feature_importances_
            names = feature_cols
        elif model_name == "lgb":
            imp = model.feature_importance(importance_type="gain")
            try:
                names = list(model.feature_name())
            except Exception:
                names = feature_cols
        else:
            pd.DataFrame({"feature": feature_cols, "importance": np.nan}).to_csv(out_csv, index=False)
            return
        n = min(len(names), len(imp))
        pd.DataFrame({"feature": names[:n], "gain": list(imp)[:n]}).sort_values("gain", ascending=False).to_csv(out_csv, index=False)
    except Exception as e:
        print(f"WARNING: failed to save feature importance for {model_name}: {e}")


def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def best_f1_threshold(y, p, grid=2001, t_min=0.001):
    thr = np.linspace(t_min, 1.0, grid)
    best_t = 0.5
    best = -1.0
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    for t in thr:
        pred = (p >= t).astype(int)
        f = float(f1_score(y, pred, zero_division=0))
        if f > best:
            best = f
            best_t = float(t)
    return best_t, best


def threshold_metrics(y, p, threshold):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / max(1, tn + fp)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def ece_score(y, p, n_bins=10):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = ((p >= lo) & (p <= hi)) if i == n_bins - 1 else ((p >= lo) & (p < hi))
        cnt = int(m.sum())
        if cnt == 0:
            rows.append({"bin": i, "bin_left": lo, "bin_right": hi, "n": 0, "mean_pred": np.nan, "frac_positive": np.nan, "abs_gap": np.nan})
            continue
        mean_pred = float(np.mean(p[m]))
        frac_pos = float(np.mean(y[m]))
        gap = abs(mean_pred - frac_pos)
        ece += (cnt / max(1, n)) * gap
        rows.append({"bin": i, "bin_left": lo, "bin_right": hi, "n": cnt, "mean_pred": mean_pred, "frac_positive": frac_pos, "abs_gap": gap})
    return float(ece), pd.DataFrame(rows)


def topk_report(y, p, fracs=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20)):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    npos = int(y.sum())
    out = {}
    if n == 0 or npos == 0:
        for f in fracs:
            tag = f"top{int(f * 100)}pct"
            out[f"{tag}_precision"] = float("nan")
            out[f"{tag}_recall"] = float("nan")
            out[f"{tag}_n"] = 0
        return out
    order = np.argsort(-p)
    for f in fracs:
        k = max(1, int(np.ceil(f * n)))
        idx = order[:k]
        tp = int(y[idx].sum())
        tag = f"top{int(f * 100)}pct"
        out[f"{tag}_precision"] = float(tp / k)
        out[f"{tag}_recall"] = float(tp / npos)
        out[f"{tag}_n"] = int(k)
    return out


def bootstrap_ci(y, p, groups=None, n_boot=1000, seed=42):
    if n_boot <= 0:
        return {}
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    rng = np.random.RandomState(seed)
    aucs, aps = [], []
    if groups is None:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            if len(np.unique(y[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y[idx], p[idx]))
            aps.append(average_precision_score(y[idx], p[idx]))
    else:
        groups = np.asarray(groups)
        uniq = pd.unique(groups)
        group_to_idx = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            sampled_g = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([group_to_idx[g] for g in sampled_g])
            if len(np.unique(y[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y[idx], p[idx]))
            aps.append(average_precision_score(y[idx], p[idx]))

    def pack(vals, prefix):
        if not vals:
            return {f"{prefix}_ci_low": float("nan"), f"{prefix}_ci_high": float("nan"), f"{prefix}_boot_sd": float("nan")}
        vals = np.asarray(vals, dtype=float)
        return {f"{prefix}_ci_low": float(np.percentile(vals, 2.5)), f"{prefix}_ci_high": float(np.percentile(vals, 97.5)), f"{prefix}_boot_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")}

    return {**pack(aucs, "auc"), **pack(aps, "ap"), "n_boot_effective": int(min(len(aucs), len(aps)))}


def save_roc_pr_curves(y, p, out_csv, dataset_name, model_name, seed, split):
    rows = []
    if len(np.unique(y)) >= 2:
        fpr, tpr, roc_thr = roc_curve(y, p)
        for a, b, t in zip(fpr, tpr, roc_thr):
            rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, "split": split, "curve": "roc", "x": float(a), "y": float(b), "threshold": float(t) if np.isfinite(t) else np.nan, "x_label": "False positive rate", "y_label": "True positive rate"})
        prec, rec, pr_thr = precision_recall_curve(y, p)
        pr_thr_full = np.concatenate([pr_thr, [np.nan]])
        for r, q, t in zip(rec, prec, pr_thr_full):
            rows.append({"dataset": dataset_name, "model": model_name, "seed": seed, "split": split, "curve": "pr", "x": float(r), "y": float(q), "threshold": float(t) if np.isfinite(t) else np.nan, "x_label": "Recall", "y_label": "Precision"})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def score_distribution_table(y, p, bins=40):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for cls in [0, 1]:
        pp = p[y == cls]
        counts, _ = np.histogram(pp, bins=edges)
        for i, cnt in enumerate(counts):
            rows.append({"class": int(cls), "bin_left": float(edges[i]), "bin_right": float(edges[i + 1]), "count": int(cnt), "density": float(cnt / max(1, len(pp)))})
    return pd.DataFrame(rows)


def save_predictions(meta_df, y_true, p_pred, groups, out_csv, dataset_name, model_name, seed, split):
    out = meta_df.copy().reset_index(drop=True)
    out["dataset"] = dataset_name
    out["model"] = model_name
    out["seed"] = int(seed)
    out["split"] = split
    out["block_group"] = np.asarray(groups).astype(str)
    out["y_true"] = np.asarray(y_true).astype(int)
    out["p_pred"] = np.asarray(p_pred).astype(float)
    out.to_csv(out_csv, index=False)
    return out


def per_cell_group_metrics(pred_df):
    if "cell_type" not in pred_df.columns:
        return pd.DataFrame()
    rows = []
    for ct, sub in pred_df.groupby("cell_type", dropna=False):
        y = sub["y_true"].to_numpy()
        p = sub["p_pred"].to_numpy()
        rows.append({
            "dataset": sub["dataset"].iloc[0],
            "model": sub["model"].iloc[0],
            "seed": int(sub["seed"].iloc[0]),
            "cell_type": str(ct),
            "n": int(len(sub)),
            "pos": int(np.sum(y)),
            "pos_rate": float(np.mean(y)),
            "auc": safe_auc(y, p),
            "ap": safe_ap(y, p),
            **topk_report(y, p, fracs=(0.01, 0.05, 0.10)),
        })
    return pd.DataFrame(rows)


def summarize_metrics(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [c for c in df.columns if c not in {"dataset", "task", "model", "seed"} and pd.api.types.is_numeric_dtype(df[c])]
    out_rows = []
    for (dataset, task, model), sub in df.groupby(["dataset", "task", "model"], dropna=False):
        r = {"dataset": dataset, "task": task, "model": model, "n_seeds": int(sub["seed"].nunique())}
        for c in metric_cols:
            vals = pd.to_numeric(sub[c], errors="coerce").dropna()
            if len(vals) == 0:
                r[f"{c}_mean"] = np.nan
                r[f"{c}_sd"] = np.nan
                r[f"{c}_sem"] = np.nan
            else:
                r[f"{c}_mean"] = float(vals.mean())
                r[f"{c}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                r[f"{c}_sem"] = float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def train_one_seed(df_raw, X, y, meta, groups, sample_weight, seed, out_dir, args):
    seed_dir = os.path.join(out_dir, f"seed_{seed}")
    ensure_dir(seed_dir)
    train_idx, val_idx, test_idx, split_diag = split_train_val_test_balanced(y, groups, seed, args.test_size, args.val_size, args.split_search_iter)
    pd.DataFrame([split_diag]).to_csv(os.path.join(seed_dir, "split_diagnostics.csv"), index=False)
    if args.save_split_assignments:
        split_rows = []
        for name, ind in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            tmp = meta.iloc[ind].copy().reset_index(drop=True)
            tmp["dataset"] = args.dataset_name
            tmp["seed"] = seed
            tmp["split"] = name
            tmp["row_index"] = ind
            tmp["block_group"] = groups[ind]
            tmp["y"] = y[ind]
            split_rows.append(tmp)
        pd.concat(split_rows, ignore_index=True).to_csv(os.path.join(seed_dir, "split_assignments.csv"), index=False)

    rows, pred_long_list, per_group_list = [], [], []
    model_names = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    for model_name in model_names:
        model_dir = os.path.join(seed_dir, model_name)
        ensure_dir(model_dir)
        train_idx_use = train_idx.copy()
        max_rows = args.max_train_rows_svm if model_name == "svm" and args.max_train_rows_svm > 0 else args.max_train_rows
        train_idx_use = maybe_subsample_train(train_idx_use, max_rows, seed=seed, y=y)

        Xtr = X.iloc[train_idx_use].copy()
        Xva = X.iloc[val_idx].copy()
        Xte = X.iloc[test_idx].copy()
        Xtr, Xva, Xte = align_frames(Xtr, Xva, Xte)
        ytr, yva, yte = y[train_idx_use], y[val_idx], y[test_idx]
        wtr = sample_weight[train_idx_use] if args.use_sample_weight else None
        wva = sample_weight[val_idx] if args.use_sample_weight else None

        print(f"\n[{args.dataset_name}] seed={seed} model={model_name.upper()} train={len(ytr):,} val={len(yva):,} test={len(yte):,} features={Xtr.shape[1]:,}")

        Xtr_fit, Xva_fit, Xte_fit = Xtr, Xva, Xte
        feature_cols_fit = list(Xtr.columns)
        if model_name == "lgb":
            Xtr_fit, Xva_fit, Xte_fit, fmap = apply_safe_feature_names(Xtr, Xva, Xte)
            fmap.to_csv(os.path.join(model_dir, "feature_name_map.csv"), index=False)
            feature_cols_fit = list(Xtr_fit.columns)

        if model_name == "rf":
            model = fit_rf(Xtr_fit, ytr, seed, args, sample_weight=wtr)
            model_path = os.path.join(model_dir, "model.joblib")
        elif model_name == "svm":
            model = fit_svm(Xtr_fit, ytr, seed, args, sample_weight=wtr)
            model_path = os.path.join(model_dir, "model.joblib")
        elif model_name == "lgb":
            model = fit_lgb(Xtr_fit, ytr, Xva_fit, yva, seed, args, sample_weight_tr=wtr, sample_weight_va=wva)
            model_path = os.path.join(model_dir, "model.txt")
        else:
            raise ValueError(f"Unsupported model: {model_name}. Supported: rf,svm,lgb")

        pva = predict_proba(model_name, model, Xva_fit)
        pte = predict_proba(model_name, model, Xte_fit)
        best_t, val_best_f1 = best_f1_threshold(yva, pva)
        test_thr = threshold_metrics(yte, pte, best_t)
        val_ece, val_cal = ece_score(yva, pva, n_bins=args.calibration_bins)
        test_ece, test_cal = ece_score(yte, pte, n_bins=args.calibration_bins)
        val_cal.to_csv(os.path.join(model_dir, "val_calibration.csv"), index=False)
        test_cal.to_csv(os.path.join(model_dir, "test_calibration.csv"), index=False)
        save_roc_pr_curves(yva, pva, os.path.join(model_dir, "val_roc_pr_curves.csv"), args.dataset_name, model_name, seed, "val")
        save_roc_pr_curves(yte, pte, os.path.join(model_dir, "test_roc_pr_curves.csv"), args.dataset_name, model_name, seed, "test")
        score_distribution_table(yte, pte, bins=args.score_bins).to_csv(os.path.join(model_dir, "test_score_distribution.csv"), index=False)
        save_model(model_name, model, model_path)
        save_feature_importance(model_name, model, feature_cols_fit, os.path.join(model_dir, "feature_importance.csv"))
        save_predictions(meta.iloc[val_idx], yva, pva, groups[val_idx], os.path.join(model_dir, "val_predictions.csv"), args.dataset_name, model_name, seed, "val")
        test_pred = save_predictions(meta.iloc[test_idx], yte, pte, groups[test_idx], os.path.join(model_dir, "test_predictions.csv"), args.dataset_name, model_name, seed, "test")
        pred_long_list.append(test_pred)
        pg = per_cell_group_metrics(test_pred)
        if not pg.empty:
            pg.to_csv(os.path.join(model_dir, "test_per_cell_group_metrics.csv"), index=False)
            per_group_list.append(pg)
        boot = bootstrap_ci(yte, pte, groups=groups[test_idx] if args.bootstrap_unit == "block" else None, n_boot=args.bootstrap_n, seed=seed + 1000)
        info = {
            "dataset": args.dataset_name,
            "task": args.task,
            "model": model_name,
            "seed": int(seed),
            "n_features": int(Xtr.shape[1]),
            "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
            "pos_train": int(ytr.sum()), "pos_val": int(yva.sum()), "pos_test": int(yte.sum()),
            "pos_rate_train": float(np.mean(ytr)), "pos_rate_val": float(np.mean(yva)), "pos_rate_test": float(np.mean(yte)),
            "val_auc": safe_auc(yva, pva), "val_ap": safe_ap(yva, pva),
            "test_auc": safe_auc(yte, pte), "test_ap": safe_ap(yte, pte),
            "val_brier": float(brier_score_loss(yva, pva)), "test_brier": float(brier_score_loss(yte, pte)),
            "val_ece": val_ece, "test_ece": test_ece,
            "val_best_threshold": best_t, "val_best_f1": val_best_f1,
            "test_f1_at_val_threshold": test_thr["f1"],
            "test_precision_at_val_threshold": test_thr["precision"],
            "test_recall_at_val_threshold": test_thr["recall"],
            "test_specificity_at_val_threshold": test_thr["specificity"],
            "test_balanced_accuracy_at_val_threshold": test_thr["balanced_accuracy"],
            "test_mcc_at_val_threshold": test_thr["mcc"],
            "test_tp": test_thr["tp"], "test_fp": test_thr["fp"], "test_tn": test_thr["tn"], "test_fn": test_thr["fn"],
            **topk_report(yte, pte), **boot,
        }
        if model_name == "lgb":
            info["best_iteration"] = int(model.best_iteration) if getattr(model, "best_iteration", None) is not None else None
        with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(info), f, indent=2)
        rows.append(info)
        print(json.dumps(to_jsonable({"model": model_name, "seed": seed, "test_auc": info["test_auc"], "test_ap": info["test_ap"], "test_f1_at_val_threshold": info["test_f1_at_val_threshold"], "auc_ci": [info.get("auc_ci_low"), info.get("auc_ci_high")], "ap_ci": [info.get("ap_ci_low"), info.get("ap_ci_high")]}), indent=2))
    return rows, pred_long_list, per_group_list


def export_figure_tables(out_dir, all_rows, pred_long, per_group):
    fig_dir = os.path.join(out_dir, "figure_tables")
    ensure_dir(fig_dir)
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(os.path.join(out_dir, "summary_by_seed.csv"), index=False)
    agg = summarize_metrics(all_rows)
    agg.to_csv(os.path.join(out_dir, "summary_agg.csv"), index=False)
    if not metrics.empty:
        fig4_cols = ["dataset", "model", "seed", "test_auc", "test_ap", "auc_ci_low", "auc_ci_high", "ap_ci_low", "ap_ci_high", "test_f1_at_val_threshold", "test_precision_at_val_threshold", "test_recall_at_val_threshold", "top1pct_precision", "top1pct_recall", "top5pct_precision", "top5pct_recall", "top10pct_precision", "top10pct_recall"]
        fig4_cols = [c for c in fig4_cols if c in metrics.columns]
        metrics[fig4_cols].to_csv(os.path.join(fig_dir, "fig4_matched_baseline_metrics_by_seed.csv"), index=False)
    if pred_long:
        pred = pd.concat(pred_long, ignore_index=True)
        pred.to_csv(os.path.join(out_dir, "all_test_predictions_long.csv"), index=False)
        pred.to_csv(os.path.join(fig_dir, "fig4_predictions_long.csv"), index=False)
        curve_rows = []
        for (dataset, model, seed), sub in pred.groupby(["dataset", "model", "seed"]):
            tmp_path = os.path.join(fig_dir, f"_tmp_curves_{dataset}_{model}_{seed}.csv")
            save_roc_pr_curves(sub["y_true"].to_numpy(), sub["p_pred"].to_numpy(), tmp_path, dataset, model, int(seed), "test")
            curve_rows.append(pd.read_csv(tmp_path))
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if curve_rows:
            pd.concat(curve_rows, ignore_index=True).to_csv(os.path.join(fig_dir, "fig4_roc_pr_curves_long.csv"), index=False)
    if per_group:
        pg = pd.concat(per_group, ignore_index=True)
        pg.to_csv(os.path.join(out_dir, "all_test_per_cell_group_metrics.csv"), index=False)
        pg.to_csv(os.path.join(fig_dir, "fig4_per_cell_group_metrics.csv"), index=False)


def build_parser():
    ap = argparse.ArgumentParser(description="Enhanced DIRECTION-style one-stage baselines for 5hmC2Stage")
    ap.add_argument("--task", required=True, choices=["site", "cell"])
    ap.add_argument("--dataset_name", default="Dataset")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", default="rf,lgb,svm", help="comma-separated: rf,svm,lgb")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--block_bp", type=int, default=10000)
    ap.add_argument("--split_search_iter", type=int, default=200)
    ap.add_argument("--max_train_rows", type=int, default=0, help="cap for RF/LGB training rows; 0 means full")
    ap.add_argument("--max_train_rows_svm", type=int, default=60000)
    ap.add_argument("--use_sample_weight", action="store_true")
    ap.add_argument("--no_cell_type_onehot", action="store_true")
    ap.add_argument("--allow_stage1_features", action="store_true", help="default false for matched one-stage baseline")
    ap.add_argument("--save_split_assignments", action="store_true")
    ap.add_argument("--bootstrap_n", type=int, default=1000)
    ap.add_argument("--bootstrap_unit", choices=["row", "block"], default="block")
    ap.add_argument("--calibration_bins", type=int, default=10)
    ap.add_argument("--score_bins", type=int, default=40)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--rf_estimators", type=int, default=800)
    ap.add_argument("--rf_max_depth", type=int, default=0)
    ap.add_argument("--rf_min_samples_leaf", type=int, default=5)
    ap.add_argument("--rf_min_samples_split", type=int, default=10)
    ap.add_argument("--rf_max_features", default="sqrt")
    ap.add_argument("--svm_c", type=float, default=2.0)
    ap.add_argument("--svm_gamma", default="scale")
    ap.add_argument("--svm_cache_size", type=int, default=4096)
    ap.add_argument("--lgb_estimators", type=int, default=2000)
    ap.add_argument("--lgb_lr", type=float, default=0.03)
    ap.add_argument("--lgb_num_leaves", type=int, default=63)
    ap.add_argument("--lgb_max_depth", type=int, default=-1)
    ap.add_argument("--lgb_min_data_in_leaf", type=int, default=50)
    ap.add_argument("--lgb_feature_fraction", type=float, default=0.85)
    ap.add_argument("--lgb_bagging_fraction", type=float, default=0.85)
    ap.add_argument("--lgb_bagging_freq", type=int, default=1)
    ap.add_argument("--lgb_lambda_l1", type=float, default=0.0)
    ap.add_argument("--lgb_lambda_l2", type=float, default=1.0)
    ap.add_argument("--lgb_early_stop", type=int, default=150)
    return ap


def main():
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    seeds = parse_seeds(args.seeds)
    print("=" * 100)
    print("Enhanced DIRECTION-style one-stage baseline")
    print("dataset:", args.dataset_name)
    print("task:", args.task)
    print("input:", args.in_csv)
    print("out_dir:", args.out_dir)
    print("models:", args.models)
    print("seeds:", seeds)
    print("block_bp:", args.block_bp)
    print("allow_stage1_features:", args.allow_stage1_features)
    print("=" * 100)
    with open(os.path.join(args.out_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(vars(args)), f, indent=2)
    df = pd.read_csv(args.in_csv, low_memory=False)
    if "chrom" not in df.columns or "pos_based" not in df.columns:
        raise ValueError("input csv must contain chrom and pos_based")
    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["pos_based"] = pd.to_numeric(df["pos_based"], errors="coerce")
    df = df[df["pos_based"].notna()].copy()
    df["pos_based"] = df["pos_based"].astype(np.int64)
    df = downcast_df_numeric(df)
    if args.task == "site":
        df_raw, X, y, feature_cols, groups, meta, sample_weight, dropped_cols = prepare_site_task(df, args)
    else:
        df_raw, X, y, feature_cols, groups, meta, sample_weight, dropped_cols = prepare_cell_task(df, args)
    ensure_dir(os.path.join(args.out_dir, "manifests"))
    pd.DataFrame({"feature": feature_cols}).to_csv(os.path.join(args.out_dir, "manifests", "feature_manifest.csv"), index=False)
    pd.DataFrame({"dropped_column": dropped_cols}).to_csv(os.path.join(args.out_dir, "manifests", "dropped_columns_for_leakage_control.csv"), index=False)
    print(f"Rows after label filtering: {len(df_raw):,}")
    print(f"Positive ratio: {np.mean(y):.4f} ({int(np.sum(y)):,}/{len(y):,})")
    print(f"Feature count: {len(feature_cols):,}")
    print(f"Number of genomic blocks: {len(pd.unique(groups)):,}")
    print(f"Sample weight used: {bool(args.use_sample_weight)}")
    all_rows, all_pred_long, all_per_group = [], [], []
    for seed in seeds:
        rows, preds, per_group = train_one_seed(df_raw, X, y, meta, groups, sample_weight, seed, args.out_dir, args)
        all_rows.extend(rows)
        all_pred_long.extend(preds)
        all_per_group.extend(per_group)
    export_figure_tables(args.out_dir, all_rows, all_pred_long, all_per_group)
    print("\nFinished.")
    print("Main outputs:")
    print(" -", os.path.join(args.out_dir, "summary_by_seed.csv"))
    print(" -", os.path.join(args.out_dir, "summary_agg.csv"))
    print(" -", os.path.join(args.out_dir, "all_test_predictions_long.csv"))
    print(" -", os.path.join(args.out_dir, "figure_tables"))


if __name__ == "__main__":
    main()
