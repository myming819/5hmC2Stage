#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P5_train_stage1_tree_global_v2.py
Upgraded Stage-1 training (site-level static): predict label_site.

New in v2:
  - keeps the original block10kb split and global site scoring logic
  - adds CatBoost and a modern residual MLP baseline (numerical tabular DL)
  - adds probability blending ensemble over trained base models
  - supports seed_list for repeated runs and aggregated summary

Notes:
  - This is still a site-level model, so the primary goal is strong ranking (AUC/AP)
  - Blend is based on validation AUC/AP and only combines already-trained base models
"""

import os, json, argparse, warnings, math
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
)

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
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def make_groups(df, block_bp=10000):
    chrom = df["chrom"].astype(str)
    pos = pd.to_numeric(df["pos_based"], errors="coerce").astype(int)
    blk = ((pos - 1) // int(block_bp)).astype(int)
    return (chrom + "|" + blk.astype(str)).astype(str).values


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


def topk_report(y, p, fracs=(0.01, 0.02, 0.05, 0.10, 0.15)):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    npos = int(y.sum())
    out = {}
    if n == 0 or npos == 0:
        for f in fracs:
            out[f"top{int(f*100)}%_R"] = float("nan")
            out[f"top{int(f*100)}%_P"] = float("nan")
        return out
    order = np.argsort(-p)
    for f in fracs:
        k = max(1, int(np.ceil(f * n)))
        idx = order[:k]
        tp = int(y[idx].sum())
        out[f"top{int(f*100)}%_P"] = tp / k
        out[f"top{int(f*100)}%_R"] = tp / npos
    return out


def best_f1_threshold(y, p, grid=2001, t_min=0.01):
    thr = np.linspace(t_min, 1.0, grid)
    best_t = 0.5
    best = -1
    for t in thr:
        pred = (p >= t).astype(int)
        f = float(f1_score(y, pred, zero_division=0))
        if f > best:
            best = f
            best_t = float(t)
    return best_t, best


def threshold_for_fixed_recall(y, p, target=0.90):
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5, float(np.mean(y)), float(np.mean(y))
    prec2, rec2 = prec[:-1], rec[:-1]
    mask = rec2 >= float(target)
    if not np.any(mask):
        j = int(np.argmax(rec2))
        return float(thr[j]), float(prec2[j]), float(rec2[j])
    idx = np.where(mask)[0]
    best = idx[np.argmax(prec2[idx])]
    return float(thr[best]), float(prec2[best]), float(rec2[best])


def save_pr(y, p, out_csv):
    prec, rec, thr = precision_recall_curve(y, p)
    thr_full = np.concatenate([thr, [np.nan]])
    pd.DataFrame({"precision": prec, "recall": rec, "threshold": thr_full}).to_csv(out_csv, index=False)


def confusion(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def save_predictions(meta_df, y_true, p_pred, out_csv):
    out = meta_df.copy()
    out["y_true"] = np.asarray(y_true).astype(int)
    out["p_pred"] = np.asarray(p_pred).astype(float)
    out.to_csv(out_csv, index=False)


def save_site_scores(meta_df, p_pred, out_csv, score_col="p_site"):
    out = meta_df[["chrom", "pos_based"]].copy()
    out[score_col] = np.asarray(p_pred).astype(float)
    out.to_csv(out_csv, index=False)


def train_xgb(Xtr, ytr, Xva, yva, seed, params):
    if xgb is None:
        raise RuntimeError("pip install xgboost")
    clf = xgb.XGBClassifier(**params, random_state=seed, n_jobs=-1, eval_metric="auc")
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf


def train_lgb(Xtr, ytr, Xva, yva, seed, params):
    if lgb is None:
        raise RuntimeError("pip install lightgbm")
    clf = lgb.LGBMClassifier(**params, random_state=seed, n_jobs=-1, objective="binary")
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=params.get("early_stopping_rounds", 200), verbose=False)])
    return clf


def train_cat(Xtr, ytr, Xva, yva, seed, params):
    if CatBoostClassifier is None:
        raise RuntimeError("pip install catboost")
    clf = CatBoostClassifier(**params, random_seed=seed, verbose=False)
    clf.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    return clf


class ResBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.BatchNorm1d(d), nn.Dropout(dropout),
            nn.Linear(d, d), nn.ReLU(), nn.BatchNorm1d(d),
        )
    def forward(self, x):
        return x + self.net(x)


class TabResMLP(nn.Module):
    def __init__(self, in_dim, d=256, depth=3, dropout=0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.BatchNorm1d(d), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResBlock(d, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d // 2, 1))
    def forward(self, x):
        h = self.inp(x)
        h = self.blocks(h)
        return self.head(h).squeeze(1)


def train_mlp(Xtr, ytr, Xva, yva, seed, params):
    if torch is None:
        raise RuntimeError("PyTorch is required for mlp")
    device = torch.device("cuda" if torch.cuda.is_available() and not params.get("cpu_only", False) else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)
    ytr_f = ytr.astype(np.float32)
    yva_f = yva.astype(np.float32)

    model = TabResMLP(Xtr_s.shape[1], d=params["hidden_dim"], depth=params["depth"], dropout=params["dropout"]).to(device)
    pos = max(1.0, float(ytr.sum()))
    neg = max(1.0, float(len(ytr) - ytr.sum()))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    ds_tr = TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr_f))
    ds_va = TensorDataset(torch.from_numpy(Xva_s), torch.from_numpy(yva_f))
    dl_tr = DataLoader(ds_tr, batch_size=params["batch_size"], shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=max(params["batch_size"], 1024), shuffle=False, drop_last=False)

    best_state, best_auc, bad = None, -1.0, 0
    for epoch in range(params["epochs"]):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
                preds.append(p)
                ys.append(yb.numpy())
        pva = np.concatenate(preds)
        yva_np = np.concatenate(ys).astype(int)
        auc = safe_auc(yva_np, pva)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= params["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.scaler_ = scaler
    return model


def predict(model_name, model, X):
    if model_name == "mlp":
        device = next(model.parameters()).device
        Xs = model.scaler_.transform(X).astype(np.float32)
        out = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(Xs), 4096):
                xb = torch.from_numpy(Xs[i:i+4096]).to(device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
                out.append(p)
        return np.concatenate(out)
    return model.predict_proba(X)[:, 1]


def save_model(model_name, model, path):
    if model_name == "xgb":
        model.save_model(path)
    elif model_name == "lgb":
        model.booster_.save_model(path)
    elif model_name == "cat":
        model.save_model(path)
    elif model_name == "mlp":
        torch.save({"state_dict": model.state_dict(), "scaler_mean": model.scaler_.mean_, "scaler_scale": model.scaler_.scale_}, path)


def feature_importance(model_name, model, feature_cols, out_csv):
    if model_name == "xgb":
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        rows = [{"feature": f, "gain": float(score.get(f"f{i}", 0.0))} for i, f in enumerate(feature_cols)]
        pd.DataFrame(rows).sort_values("gain", ascending=False).to_csv(out_csv, index=False)
    elif model_name == "lgb":
        imp = model.booster_.feature_importance(importance_type="gain")
        pd.DataFrame({"feature": feature_cols, "gain": imp}).sort_values("gain", ascending=False).to_csv(out_csv, index=False)
    elif model_name == "cat":
        imp = model.get_feature_importance()
        pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False).to_csv(out_csv, index=False)
    elif model_name == "mlp":
        pd.DataFrame({"feature": feature_cols, "importance": np.nan}).to_csv(out_csv, index=False)


def prepare_score_matrix(score_df, feature_cols):
    score_df = score_df.copy()
    score_df["chrom"] = score_df["chrom"].astype(str).str.strip()
    score_df["pos_based"] = pd.to_numeric(score_df["pos_based"], errors="coerce")
    score_df = score_df.dropna(subset=["pos_based"]).copy()
    score_df["pos_based"] = score_df["pos_based"].astype(int)
    score_df = score_df.drop_duplicates(["chrom", "pos_based"], keep="first")
    non_feature_drop = {"label_site", "site_role", "pos_count", "obs_count", "pos_ratio", "label_cell", "gene_id", "gene_name", "cell_type", "y_true", "p_pred", "p_site", "p_site_stage1"}
    score_X = score_df.drop(columns=[c for c in score_df.columns if c in non_feature_drop], errors="ignore").copy()
    for c in score_X.columns:
        if c not in ["chrom", "pos_based"] and not pd.api.types.is_numeric_dtype(score_X[c]):
            score_X[c] = pd.to_numeric(score_X[c], errors="coerce")
    out = pd.DataFrame(index=score_X.index)
    for c in feature_cols:
        out[c] = score_X[c] if c in score_X.columns else np.nan
    return score_df[["chrom", "pos_based"]].reset_index(drop=True), out


def blend_weights_from_val(yva, pred_dict):
    rows = []
    for name, p in pred_dict.items():
        auc = safe_auc(yva, p)
        ap = safe_ap(yva, p)
        score = max(0.0, auc - 0.5) + max(0.0, ap - yva.mean())
        rows.append((name, auc, ap, score))
    pdf = pd.DataFrame(rows, columns=["model", "val_auc", "val_ap", "blend_score"])
    s = pdf["blend_score"].to_numpy()
    if np.all(s <= 0):
        w = np.ones(len(pdf), dtype=float) / len(pdf)
    else:
        w = s / s.sum()
    return {m: float(wi) for m, wi in zip(pdf["model"], w)}, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=r"model_input/PBMC_CPG_ALL_UNION/stage1_site_static_union.csv")
    ap.add_argument("--out_dir", default=r"model_output/stage1_PBMC_v2")
    ap.add_argument("--models", default="xgb,lgb,cat,mlp,blend")
    ap.add_argument("--block_bp", type=int, default=10000)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_list", default="")
    ap.add_argument("--score_csv", default=r"model_input/PBMC_CPG_ALL_UNION/stage2_cell_static_dynamic_union.csv")
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
    ap.add_argument("--lgb_early_stop", type=int, default=300)
    # cat
    ap.add_argument("--cat_iters", type=int, default=5000)
    ap.add_argument("--cat_lr", type=float, default=0.03)
    ap.add_argument("--cat_depth", type=int, default=8)
    ap.add_argument("--cat_l2", type=float, default=6.0)
    # mlp
    ap.add_argument("--mlp_hidden_dim", type=int, default=256)
    ap.add_argument("--mlp_depth", type=int, default=3)
    ap.add_argument("--mlp_dropout", type=float, default=0.15)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    ap.add_argument("--mlp_batch_size", type=int, default=1024)
    ap.add_argument("--mlp_epochs", type=int, default=100)
    ap.add_argument("--mlp_patience", type=int, default=10)
    ap.add_argument("--target_recall", type=float, default=0.90)
    args = ap.parse_args()
    ensure_dir(args.out_dir)

    seed_list = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()] if args.seed_list.strip() else [args.seed]
    all_rows = []
    for seed in seed_list:
        df = pd.read_csv(args.in_csv, low_memory=False)
        req = ["chrom", "pos_based", "label_site"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"Missing required columns: {miss}")
        y = pd.to_numeric(df["label_site"], errors="coerce")
        keep = y.notna()
        df = df.loc[keep].copy()
        y = y.loc[keep].astype(int).to_numpy()
        df["chrom"] = df["chrom"].astype(str).str.strip()
        df["pos_based"] = pd.to_numeric(df["pos_based"], errors="coerce").astype(int)
        drop_cols = {"chrom", "pos_based", "label_site", "site_role", "pos_count", "obs_count", "pos_ratio", "pos_conf_sum", "neg_conf_sum", "obs_conf_sum", "pos_conf_ratio", "site_soft_score", "gene_id", "gene_name"}
        X = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore").copy()
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        feature_cols = list(X.columns)
        groups = make_groups(df, block_bp=args.block_bp)
        idx = np.arange(len(df))
        gss1 = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        trainval_idx, test_idx = next(gss1.split(idx, y, groups))
        groups_tv = groups[trainval_idx]
        y_tv = y[trainval_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_size / (1 - args.test_size), random_state=seed + 1)
        tr_rel, va_rel = next(gss2.split(trainval_idx, y_tv, groups_tv))
        train_idx = trainval_idx[tr_rel]
        val_idx = trainval_idx[va_rel]
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X.iloc[train_idx])
        Xva = imp.transform(X.iloc[val_idx])
        Xte = imp.transform(X.iloc[test_idx])
        Xall = imp.transform(X)
        ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]
        spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
        meta_test = df.iloc[test_idx][["chrom", "pos_based"]].reset_index(drop=True)
        meta_all = df[["chrom", "pos_based"]].reset_index(drop=True)
        score_meta, Xscore = None, None
        if args.score_csv:
            score_df = pd.read_csv(args.score_csv, low_memory=False)
            score_meta, Xscore_df = prepare_score_matrix(score_df, feature_cols)
            Xscore = imp.transform(Xscore_df)
        print("=" * 100)
        print(f"🌲 Stage-1 v2 | seed={seed}")
        print(f"Input: {args.in_csv}")
        print(f"N={len(df):,} features={len(feature_cols)} pos={int(y.sum()):,} ({y.mean():.2%})")
        print(f"Split: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,} | spw={spw:.3f}")
        base_models = [m for m in [x.strip().lower() for x in args.models.split(",") if x.strip()] if m != "blend"]
        val_preds, test_preds, all_preds, score_preds = {}, {}, {}, {}
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        ensure_dir(seed_dir)
        for m in base_models:
            mdir = os.path.join(seed_dir, m)
            ensure_dir(mdir)
            if m == "xgb":
                params = dict(n_estimators=args.xgb_estimators, learning_rate=args.xgb_lr, max_depth=args.xgb_depth, subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample, reg_lambda=args.xgb_lambda, reg_alpha=args.xgb_alpha, gamma=args.xgb_gamma, min_child_weight=args.xgb_min_child_weight, scale_pos_weight=spw, tree_method=args.xgb_tree_method)
                model = train_xgb(Xtr, ytr, Xva, yva, seed + 10, params)
                model_path = os.path.join(mdir, "model.xgb.json")
            elif m == "lgb":
                params = dict(n_estimators=args.lgb_estimators, learning_rate=args.lgb_lr, num_leaves=args.lgb_num_leaves, min_data_in_leaf=args.lgb_min_data_in_leaf, feature_fraction=args.lgb_feature_fraction, bagging_fraction=args.lgb_bagging_fraction, bagging_freq=args.lgb_bagging_freq, scale_pos_weight=spw, early_stopping_rounds=args.lgb_early_stop)
                model = train_lgb(Xtr, ytr, Xva, yva, seed + 20, params)
                model_path = os.path.join(mdir, "model.lgb.txt")
            elif m == "cat":
                params = dict(iterations=args.cat_iters, learning_rate=args.cat_lr, depth=args.cat_depth, l2_leaf_reg=args.cat_l2, loss_function="Logloss", eval_metric="AUC", class_weights=[1.0, float(spw)])
                model = train_cat(Xtr, ytr, Xva, yva, seed + 30, params)
                model_path = os.path.join(mdir, "model.cat.cbm")
            elif m == "mlp":
                params = dict(hidden_dim=args.mlp_hidden_dim, depth=args.mlp_depth, dropout=args.mlp_dropout, lr=args.mlp_lr, weight_decay=args.mlp_weight_decay, batch_size=args.mlp_batch_size, epochs=args.mlp_epochs, patience=args.mlp_patience)
                model = train_mlp(Xtr, ytr, Xva, yva, seed + 40, params)
                model_path = os.path.join(mdir, "model.mlp.pt")
            else:
                continue
            pva = predict(m, model, Xva)
            pte = predict(m, model, Xte)
            pall = predict(m, model, Xall)
            val_preds[m], test_preds[m], all_preds[m] = pva, pte, pall
            if Xscore is not None:
                score_preds[m] = predict(m, model, Xscore)
            va_auc, va_ap = safe_auc(yva, pva), safe_ap(yva, pva)
            te_auc, te_ap = safe_auc(yte, pte), safe_ap(yte, pte)
            te_top = topk_report(yte, pte)
            best_t, best_f1 = best_f1_threshold(yva, pva)
            save_pr(yte, pte, os.path.join(mdir, "pr_curve.csv"))
            save_predictions(meta_test, yte, pte, os.path.join(mdir, "test_predictions.csv"))
            save_site_scores(meta_all, pall, os.path.join(mdir, "site_scores_train_df.csv"))
            if Xscore is not None:
                save_site_scores(score_meta, score_preds[m], os.path.join(mdir, "global_site_scores.csv"))
            save_model(m, model, model_path)
            feature_importance(m, model, feature_cols, os.path.join(mdir, "feature_importance.csv"))
            all_rows.append({"seed": seed, "model": m, "val_auc": va_auc, "val_ap": va_ap, "test_auc": te_auc, "test_ap": te_ap, "test_top5_R": te_top["top5%_R"], "test_f1_bestT": best_f1, "test_rec_bestT": float(recall_score(yte, (pte >= best_t).astype(int), zero_division=0)), "test_prec_bestT": float(precision_score(yte, (pte >= best_t).astype(int), zero_division=0))})
            print(f"[{m.upper()}] VAL AUC={va_auc:.4f} AP={va_ap:.4f} | TEST AUC={te_auc:.4f} AP={te_ap:.4f}")
        if "blend" in [x.strip().lower() for x in args.models.split(",") if x.strip()] and len(val_preds) >= 2:
            weights, blend_df = blend_weights_from_val(yva, val_preds)
            blend_pva = sum(weights[m] * val_preds[m] for m in weights)
            blend_pte = sum(weights[m] * test_preds[m] for m in weights)
            blend_all = sum(weights[m] * all_preds[m] for m in weights)
            va_auc, va_ap = safe_auc(yva, blend_pva), safe_ap(yva, blend_pva)
            te_auc, te_ap = safe_auc(yte, blend_pte), safe_ap(yte, blend_pte)
            te_top = topk_report(yte, blend_pte)
            best_t, best_f1 = best_f1_threshold(yva, blend_pva)
            mdir = os.path.join(seed_dir, "blend")
            ensure_dir(mdir)
            save_pr(yte, blend_pte, os.path.join(mdir, "pr_curve.csv"))
            save_predictions(meta_test, yte, blend_pte, os.path.join(mdir, "test_predictions.csv"))
            save_site_scores(meta_all, blend_all, os.path.join(mdir, "site_scores_train_df.csv"))
            if Xscore is not None and len(score_preds) >= 2:
                blend_score = sum(weights[m] * score_preds[m] for m in weights if m in score_preds)
                save_site_scores(score_meta, blend_score, os.path.join(mdir, "global_site_scores.csv"))
            blend_df["weight"] = blend_df["model"].map(weights)
            blend_df.to_csv(os.path.join(mdir, "blend_weights.csv"), index=False)
            all_rows.append({"seed": seed, "model": "blend", "val_auc": va_auc, "val_ap": va_ap, "test_auc": te_auc, "test_ap": te_ap, "test_top5_R": te_top["top5%_R"], "test_f1_bestT": best_f1, "test_rec_bestT": float(recall_score(yte, (blend_pte >= best_t).astype(int), zero_division=0)), "test_prec_bestT": float(precision_score(yte, (blend_pte >= best_t).astype(int), zero_division=0))})
            print(f"[BLEND] VAL AUC={va_auc:.4f} AP={va_ap:.4f} | TEST AUC={te_auc:.4f} AP={te_ap:.4f}")
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(args.out_dir, "summary_all_seeds.csv"), index=False)
    agg = summary_df.groupby("model")[["val_auc", "val_ap", "test_auc", "test_ap", "test_top5_R", "test_f1_bestT"]].agg(["mean", "std", "min", "max"])
    agg.to_csv(os.path.join(args.out_dir, "summary_agg.csv"))
    print("\n✅ Stage-1 v2 finished. Summary:", os.path.join(args.out_dir, "summary_agg.csv"))
    print(agg.to_string())


if __name__ == "__main__":
    main()
