#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P5_make_stage1_oof_scores_enhanced.py
Build block-level out-of-fold Stage-1 Propensity Scores.

Main improvements:
  - OOF scores are generated at genomic-block level to avoid Stage-1/Stage-2 split leakage.
  - Supports xgb/lgb/cat base learners and weighted blend.
  - Exports base-model OOF scores and a final p_site score for Stage-2.

Output:
  - stage1_oof_site_scores.csv: chrom,pos_based,p_site,p_site_xgb,p_site_lgb,p_site_cat,oof_fold
  - stage1_oof_metrics.csv: fold-level OOF metrics on labeled Stage-1 sites
  - stage1_oof_summary.csv: overall OOF AUC/AP for each score column
"""

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


def ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)


def make_groups(df, block_bp=10000):
    chrom = df["chrom"].astype(str).str.strip()
    pos = pd.to_numeric(df["pos_based"], errors="coerce").astype(int)
    blk = ((pos - 1) // int(block_bp)).astype(int)
    return (chrom + "|" + blk.astype(str)).astype(str).values


def safe_auc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_ap(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def prepare_stage1_features(df):
    df = df.copy()
    y = pd.to_numeric(df["label_site"], errors="coerce")
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int).to_numpy()
    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["pos_based"] = pd.to_numeric(df["pos_based"], errors="coerce").astype(int)
    drop_cols = {
        "chrom", "pos_based", "label_site", "site_role",
        "pos_count", "obs_count", "pos_ratio", "pos_conf_sum", "neg_conf_sum",
        "obs_conf_sum", "pos_conf_ratio", "site_soft_score",
        "gene_id", "gene_name", "label_cell", "cell_type", "sample_weight",
        "y_true", "p_pred", "p_site", "p_site_stage1"
    }
    X = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore").copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    bad_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if bad_cols:
        X = X.drop(columns=bad_cols)
    return df, X, y, list(X.columns)


def prepare_score_matrix(score_df, feature_cols):
    score_df = score_df.copy()
    score_df["chrom"] = score_df["chrom"].astype(str).str.strip()
    score_df["pos_based"] = pd.to_numeric(score_df["pos_based"], errors="coerce")
    score_df = score_df.dropna(subset=["pos_based"]).copy()
    score_df["pos_based"] = score_df["pos_based"].astype(int)
    score_df = score_df.drop_duplicates(["chrom", "pos_based"], keep="first").reset_index(drop=True)

    non_feature_drop = {
        "chrom", "pos_based", "label_site", "site_role", "pos_count", "obs_count", "pos_ratio",
        "pos_conf_sum", "neg_conf_sum", "obs_conf_sum", "pos_conf_ratio", "site_soft_score",
        "label_cell", "cell_type", "gene_id", "gene_name", "sample_weight",
        "y_true", "p_pred", "p_site", "p_site_stage1"
    }
    raw = score_df.drop(columns=[c for c in score_df.columns if c in non_feature_drop], errors="ignore").copy()
    for c in raw.columns:
        if not pd.api.types.is_numeric_dtype(raw[c]):
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    Xscore = pd.DataFrame(index=score_df.index)
    for c in feature_cols:
        Xscore[c] = raw[c] if c in raw.columns else np.nan
    meta = score_df[["chrom", "pos_based"]].copy()
    return meta, Xscore


def train_xgb(Xtr, ytr, Xva, yva, seed, args):
    if xgb is None:
        raise RuntimeError("xgboost is not installed")
    neg = max(1.0, float(len(ytr) - ytr.sum()))
    pos = max(1.0, float(ytr.sum()))
    clf = xgb.XGBClassifier(
        n_estimators=args.xgb_estimators,
        learning_rate=args.xgb_lr,
        max_depth=args.xgb_depth,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample,
        reg_lambda=args.xgb_lambda,
        reg_alpha=args.xgb_alpha,
        gamma=args.xgb_gamma,
        min_child_weight=args.xgb_min_child_weight,
        scale_pos_weight=neg / pos,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method=args.xgb_tree_method,
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf


def train_lgb(Xtr, ytr, Xva, yva, seed, args):
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    neg = max(1.0, float(len(ytr) - ytr.sum()))
    pos = max(1.0, float(ytr.sum()))
    clf = lgb.LGBMClassifier(
        n_estimators=args.lgb_estimators,
        learning_rate=args.lgb_lr,
        num_leaves=args.lgb_num_leaves,
        min_data_in_leaf=args.lgb_min_data_in_leaf,
        feature_fraction=args.lgb_feature_fraction,
        bagging_fraction=args.lgb_bagging_fraction,
        bagging_freq=args.lgb_bagging_freq,
        lambda_l2=args.lgb_lambda_l2,
        scale_pos_weight=neg / pos,
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=args.lgb_early_stop, verbose=False)]
    )
    return clf


def train_cat(Xtr, ytr, Xva, yva, seed, args):
    if CatBoostClassifier is None:
        raise RuntimeError("catboost is not installed")
    neg = max(1.0, float(len(ytr) - ytr.sum()))
    pos = max(1.0, float(ytr.sum()))
    model = CatBoostClassifier(
        iterations=args.cat_iters,
        learning_rate=args.cat_lr,
        depth=args.cat_depth,
        l2_leaf_reg=args.cat_l2,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1.0, float(neg / pos)],
        od_type="Iter",
        od_wait=args.cat_early_stop,
        random_seed=seed,
        verbose=False,
    )
    model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    return model


def predict_model(name, model, X):
    if name == "lgb":
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


def blend_weights_from_val(yva, pred_dict):
    rows = []
    base_rate = float(np.mean(yva))
    for name, p in pred_dict.items():
        auc = safe_auc(yva, p)
        ap = safe_ap(yva, p)
        score = max(0.0, auc - 0.5) + max(0.0, ap - base_rate)
        rows.append((name, auc, ap, score))
    df = pd.DataFrame(rows, columns=["model", "val_auc", "val_ap", "blend_score"])
    s = df["blend_score"].to_numpy(dtype=float)
    if len(s) == 0:
        return {}, df
    if np.all(s <= 0) or not np.all(np.isfinite(s)):
        w = np.ones(len(df), dtype=float) / len(df)
    else:
        w = s / s.sum()
    weights = {m: float(wi) for m, wi in zip(df["model"], w)}
    return weights, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_csv", required=True)
    ap.add_argument("--score_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--models", default="cat,lgb,xgb,blend")
    ap.add_argument("--score_output", choices=["blend", "cat", "lgb", "xgb", "mean"], default="blend")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block_bp", type=int, default=10000)
    ap.add_argument("--inner_val_size", type=float, default=0.15)
    # xgb
    ap.add_argument("--xgb_tree_method", default="hist")
    ap.add_argument("--xgb_estimators", type=int, default=2500)
    ap.add_argument("--xgb_lr", type=float, default=0.03)
    ap.add_argument("--xgb_depth", type=int, default=6)
    ap.add_argument("--xgb_subsample", type=float, default=0.8)
    ap.add_argument("--xgb_colsample", type=float, default=0.8)
    ap.add_argument("--xgb_lambda", type=float, default=2.0)
    ap.add_argument("--xgb_alpha", type=float, default=0.1)
    ap.add_argument("--xgb_gamma", type=float, default=0.5)
    ap.add_argument("--xgb_min_child_weight", type=float, default=4.0)
    # lgb
    ap.add_argument("--lgb_estimators", type=int, default=10000)
    ap.add_argument("--lgb_lr", type=float, default=0.03)
    ap.add_argument("--lgb_num_leaves", type=int, default=63)
    ap.add_argument("--lgb_min_data_in_leaf", type=int, default=40)
    ap.add_argument("--lgb_feature_fraction", type=float, default=0.8)
    ap.add_argument("--lgb_bagging_fraction", type=float, default=0.8)
    ap.add_argument("--lgb_bagging_freq", type=int, default=1)
    ap.add_argument("--lgb_lambda_l2", type=float, default=1.0)
    ap.add_argument("--lgb_early_stop", type=int, default=300)
    # cat
    ap.add_argument("--cat_iters", type=int, default=5000)
    ap.add_argument("--cat_lr", type=float, default=0.03)
    ap.add_argument("--cat_depth", type=int, default=8)
    ap.add_argument("--cat_l2", type=float, default=6.0)
    ap.add_argument("--cat_early_stop", type=int, default=300)
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir else os.path.dirname(args.out_csv)
    ensure_dir(out_dir)
    ensure_dir(os.path.dirname(args.out_csv))

    base_models = [m.strip().lower() for m in args.models.split(",") if m.strip() and m.strip().lower() != "blend"]
    use_blend = "blend" in [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if not base_models:
        raise ValueError("At least one base model is required")

    print("=" * 100)
    print("Build block-level OOF Stage-1 Propensity Scores")
    print("Stage-1 CSV:", args.stage1_csv)
    print("Score CSV:", args.score_csv)
    print("Output:", args.out_csv)
    print("Models:", base_models, "| blend:", use_blend, "| score_output:", args.score_output)
    print("=" * 100)

    stage1_raw = pd.read_csv(args.stage1_csv, low_memory=False)
    req = {"chrom", "pos_based", "label_site"}
    miss = req - set(stage1_raw.columns)
    if miss:
        raise ValueError(f"stage1_csv missing columns: {miss}")
    stage1_df, X, y, feature_cols = prepare_stage1_features(stage1_raw)
    stage1_groups = make_groups(stage1_df, args.block_bp)

    score_raw = pd.read_csv(args.score_csv, low_memory=False)
    score_meta, Xscore = prepare_score_matrix(score_raw, feature_cols)
    score_groups = make_groups(score_meta, args.block_bp)

    unique_score_groups = pd.Series(score_groups).drop_duplicates().reset_index(drop=True)
    dummy_y = np.arange(len(unique_score_groups)) % 2
    try:
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(unique_score_groups, dummy_y, unique_score_groups)
        splitter_name = "StratifiedGroupKFold"
    except Exception:
        splitter = GroupKFold(n_splits=args.folds)
        split_iter = splitter.split(unique_score_groups, dummy_y, unique_score_groups)
        splitter_name = "GroupKFold"

    print(f"Fold splitter: {splitter_name} | folds={args.folds} | block_bp={args.block_bp}")
    print(f"Stage-1 labeled sites: {len(stage1_df):,} | score sites: {len(score_meta):,} | features={len(feature_cols):,}")

    pred_cols = {m: np.full(len(score_meta), np.nan, dtype=float) for m in base_models}
    pred_cols["mean"] = np.full(len(score_meta), np.nan, dtype=float)
    pred_cols["blend"] = np.full(len(score_meta), np.nan, dtype=float)
    fold_id = np.full(len(score_meta), -1, dtype=int)
    metric_rows = []
    weight_rows = []
    group_array = unique_score_groups.to_numpy()

    for k, (_, held_group_idx) in enumerate(split_iter, start=1):
        held_groups = set(group_array[held_group_idx])
        train_mask_all = ~pd.Series(stage1_groups).isin(held_groups).to_numpy()
        held_score_mask = pd.Series(score_groups).isin(held_groups).to_numpy()
        train_idx_all = np.where(train_mask_all)[0]
        y_train_all = y[train_idx_all]
        groups_train_all = stage1_groups[train_idx_all]

        inner = GroupShuffleSplit(n_splits=1, test_size=args.inner_val_size, random_state=args.seed + k)
        tr_rel, va_rel = next(inner.split(train_idx_all, y_train_all, groups_train_all))
        tr_idx = train_idx_all[tr_rel]
        va_idx = train_idx_all[va_rel]

        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X.iloc[tr_idx])
        Xva = imp.transform(X.iloc[va_idx])
        Xheld = imp.transform(Xscore.iloc[held_score_mask])
        ytr, yva = y[tr_idx], y[va_idx]

        val_preds = {}
        held_preds = {}
        for m in base_models:
            print(f"Fold {k} | training {m.upper()} | train={len(tr_idx):,} val={len(va_idx):,} held_score={held_score_mask.sum():,}")
            if m == "xgb":
                model = train_xgb(Xtr, ytr, Xva, yva, args.seed + 100*k + 1, args)
            elif m == "lgb":
                model = train_lgb(Xtr, ytr, Xva, yva, args.seed + 100*k + 2, args)
            elif m == "cat":
                model = train_cat(Xtr, ytr, Xva, yva, args.seed + 100*k + 3, args)
            else:
                raise ValueError(f"Unsupported model: {m}")
            val_preds[m] = predict_model(m, model, Xva)
            held_preds[m] = predict_model(m, model, Xheld)
            pred_cols[m][held_score_mask] = held_preds[m]

        base_stack = np.vstack([held_preds[m] for m in base_models])
        pred_cols["mean"][held_score_mask] = np.nanmean(base_stack, axis=0)

        if use_blend and len(base_models) >= 2:
            weights, wdf = blend_weights_from_val(yva, val_preds)
        else:
            weights = {m: 1.0 / len(base_models) for m in base_models}
            wdf = pd.DataFrame({"model": list(weights), "val_auc": np.nan, "val_ap": np.nan, "blend_score": np.nan, "weight": list(weights.values())})
        for m, wt in weights.items():
            weight_rows.append({"fold": k, "model": m, "weight": wt})
        pred_cols["blend"][held_score_mask] = sum(weights[m] * held_preds[m] for m in weights)
        fold_id[held_score_mask] = k

        held_meta = score_meta.loc[held_score_mask, ["chrom", "pos_based"]].copy()
        labeled_held = held_meta.merge(stage1_df[["chrom", "pos_based", "label_site"]], on=["chrom", "pos_based"], how="inner")
        tmp = held_meta.copy()
        for name in list(pred_cols.keys()):
            tmp[f"p_site_{name}"] = pred_cols[name][held_score_mask]
        labeled_pred = labeled_held.merge(tmp, on=["chrom", "pos_based"], how="left")

        row_base = {
            "fold": k,
            "train_labeled_sites": int(len(tr_idx)),
            "val_labeled_sites": int(len(va_idx)),
            "heldout_score_sites": int(held_score_mask.sum()),
            "heldout_labeled_sites": int(len(labeled_pred)),
            "heldout_labeled_pos": int(labeled_pred["label_site"].sum()) if len(labeled_pred) else 0,
        }
        for name in list(pred_cols.keys()):
            col = f"p_site_{name}"
            row_base[f"{name}_auc"] = safe_auc(labeled_pred["label_site"], labeled_pred[col]) if len(labeled_pred) else np.nan
            row_base[f"{name}_ap"] = safe_ap(labeled_pred["label_site"], labeled_pred[col]) if len(labeled_pred) else np.nan
        metric_rows.append(row_base)
        print(f"Fold {k}: heldout_labeled={len(labeled_pred):,} blend_AUC={row_base.get('blend_auc', np.nan):.4f} blend_AP={row_base.get('blend_ap', np.nan):.4f}")

    out = score_meta.copy()
    for m in base_models:
        out[f"p_site_{m}"] = pred_cols[m]
    out["p_site_mean"] = pred_cols["mean"]
    out["p_site_blend"] = pred_cols["blend"]

    if args.score_output in base_models:
        out["p_site"] = out[f"p_site_{args.score_output}"]
    elif args.score_output == "mean":
        out["p_site"] = out["p_site_mean"]
    else:
        out["p_site"] = out["p_site_blend"] if use_blend else out["p_site_mean"]
    out["oof_fold"] = fold_id

    if out["p_site"].isna().any():
        raise RuntimeError(f"Missing final p_site scores: {int(out['p_site'].isna().sum())}")
    out.to_csv(args.out_csv, index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(out_dir, "stage1_oof_metrics.csv"), index=False)
    pd.DataFrame(weight_rows).to_csv(os.path.join(out_dir, "stage1_oof_blend_weights.csv"), index=False)

    labeled_all = stage1_df[["chrom", "pos_based", "label_site"]].merge(out, on=["chrom", "pos_based"], how="inner")
    labeled_all.to_csv(os.path.join(out_dir, "stage1_oof_labeled_predictions.csv"), index=False)

    summary_rows = []
    score_columns = ["p_site"] + [f"p_site_{m}" for m in base_models] + ["p_site_mean", "p_site_blend"]
    for col in score_columns:
        if col in labeled_all.columns:
            summary_rows.append({
                "score_col": col,
                "n_labeled_scored": int(len(labeled_all)),
                "oof_auc": safe_auc(labeled_all["label_site"], labeled_all[col]),
                "oof_ap": safe_ap(labeled_all["label_site"], labeled_all[col]),
                "n_score_sites": int(len(out)),
                "score_coverage": float(out[col].notna().mean()) if col in out.columns else np.nan,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "stage1_oof_summary.csv"), index=False)

    print("=" * 100)
    print("Saved OOF scores:", args.out_csv)
    print(summary.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
