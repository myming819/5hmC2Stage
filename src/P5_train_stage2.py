#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P5_train_stage2_tree_refine_v2.py
Upgraded Stage-2 training with stronger model options on a fixed dataset.

New in v2:
  - keeps scientifically valid block10kb split, hard gating, soft prior and sample_weight
  - adds CatBoost and a modern residual MLP baseline for tabular data
  - adds blend ensemble over selected base models
  - preserves LOCTO and multi-seed evaluation
"""

import os, argparse, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

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


def save_pr(y, p, out_csv):
    prec, rec, thr = precision_recall_curve(y, p)
    thr_full = np.concatenate([thr, [np.nan]])
    pd.DataFrame({"precision": prec, "recall": rec, "threshold": thr_full}).to_csv(out_csv, index=False)


def save_fold_predictions(meta_df, y, p, out_csv):
    out = meta_df.copy()
    out["y_true"] = np.asarray(y).astype(int)
    out["p_pred"] = np.asarray(p).astype(float)
    out.to_csv(out_csv, index=False)


def downcast_df_numeric(df):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="float")
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], downcast="integer")
    return out


def feature_importance(model_name, model, feature_cols, out_csv):
    try:
        if model_name == "xgb":
            imp = model.feature_importances_
            names = list(feature_cols)
        elif model_name == "lgb":
            imp = model.feature_importance(importance_type="gain")
            # Keep original feature names. If LightGBM is trained on a numpy array,
            # model.feature_name() becomes Column_0/Column_1/..., which is not useful.
            names = list(feature_cols)
        elif model_name == "cat":
            imp = model.get_feature_importance()
            try:
                names = list(model.feature_names_)
            except Exception:
                names = list(feature_cols)
        else:
            pd.DataFrame({"feature": feature_cols, "importance": np.nan}).to_csv(out_csv, index=False)
            return
        n = min(len(names), len(imp))
        pd.DataFrame({"feature": names[:n], "gain": list(imp)[:n]}).sort_values("gain", ascending=False).to_csv(out_csv, index=False)
    except Exception as e:
        print(f"⚠️ Failed to save feature importance for {model_name}: {e}")


def load_stage1_scores(stage1_scores_csv, score_col="p_site"):
    scores = pd.read_csv(stage1_scores_csv, low_memory=False)
    need = {"chrom", "pos_based", score_col}
    if not need.issubset(scores.columns):
        raise ValueError(f"stage1_scores_csv must contain chrom,pos_based,{score_col}")
    scores["chrom"] = scores["chrom"].astype(str).str.strip()
    scores["pos_based"] = pd.to_numeric(scores["pos_based"], errors="coerce").astype("Int64")
    scores[score_col] = pd.to_numeric(scores[score_col], errors="coerce")
    scores = scores.dropna(subset=["pos_based", score_col]).copy()
    scores["pos_based"] = scores["pos_based"].astype(int)
    scores = scores.drop_duplicates(["chrom", "pos_based"])
    return scores[["chrom", "pos_based", score_col]].copy()


def report_stage1_score_coverage(df, scores, score_col="p_site"):
    site_df = df[["chrom", "pos_based"]].drop_duplicates()
    site_sc = scores[["chrom", "pos_based"]].drop_duplicates()
    merged = site_df.merge(site_sc, on=["chrom", "pos_based"], how="left", indicator=True)
    covered_sites = int((merged["_merge"] == "both").sum())
    total_sites = int(len(site_df))
    row_cov = df.merge(scores[["chrom", "pos_based", score_col]], on=["chrom", "pos_based"], how="left")
    covered_rows = int(row_cov[score_col].notna().sum())
    total_rows = int(len(df))
    print(f"📊 Stage1 score coverage | sites: {covered_sites:,}/{total_sites:,} ({covered_sites/max(1,total_sites):.2%}) | rows: {covered_rows:,}/{total_rows:,} ({covered_rows/max(1,total_rows):.2%})")


def apply_candidate_gating(df, scores, top_frac, score_col="p_site"):
    uniq = scores.sort_values(score_col, ascending=False).reset_index(drop=True)
    k = max(1, int(np.ceil(float(top_frac) * len(uniq))))
    keep = uniq.iloc[:k][["chrom", "pos_based"]]
    out = df.merge(keep, on=["chrom", "pos_based"], how="inner")
    thr = float(uniq.iloc[k - 1][score_col])
    kept_sites = int(len(keep))
    out_sites = int(out[["chrom", "pos_based"]].drop_duplicates().shape[0])
    return out, thr, kept_sites, out_sites


def add_stage1_soft_feature(df, scores, score_col="p_site"):
    tmp = scores[["chrom", "pos_based", score_col]].rename(columns={score_col: "p_site_stage1"})
    return df.merge(tmp, on=["chrom", "pos_based"], how="left")


def add_stage1_score_transforms(df, eps=1e-6):
    """
    Add monotonic transformations of p_site_stage1.
    These are safe because they use only the OOF/global Stage-1 score already merged by site.
    """
    if "p_site_stage1" not in df.columns:
        return df
    out = df.copy()
    p = pd.to_numeric(out["p_site_stage1"], errors="coerce").astype("float32")
    p_clip = p.clip(eps, 1.0 - eps)
    out["p_site_stage1_logit"] = np.log(p_clip / (1.0 - p_clip)).astype("float32")
    out["p_site_stage1_rank"] = p.rank(method="average", pct=True).astype("float32")
    out["p_site_stage1_centered"] = (p - p.mean()).astype("float32")
    out["p_site_stage1_sq"] = (p * p).astype("float32")
    return out


def add_stage1_score_interactions(df):
    """
    Couple Stage-1 site propensity with dynamic 5mC features.
    This implements the biological idea that a site's basal 5hmC propensity may be modulated
    by cell-group-associated site/promoter 5mC states.
    """
    if "p_site_stage1" not in df.columns:
        return df
    out = df.copy()
    p = pd.to_numeric(out["p_site_stage1"], errors="coerce").astype("float32")
    candidate_cols = [
        "mC_site_this_cell", "mC_site_mean", "mC_site_std", "mC_site_range",
        "mC_site_delta", "mC_site_z", "mC_site_range_over_mean", "mC_site_top2gap_over_std",
        "mC_promoter_this_cell", "mC_promoter_mean_cells", "mC_promoter_std_cells",
        "mC_promoter_delta", "mC_promoter_z", "mC_promoter_rank_pct",
        "mC_site_minus_promoter", "mC_site_over_promoter", "mC_site_z_minus_promoter_z",
        "mC_site_delta_x_promoter_rank", "mC_site_this_over_mean", "mC_promoter_this_over_mean",
        "cell_mc_mean", "cell_mc_std", "cell_mc_q90"
    ]
    added = 0
    for c in candidate_cols:
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce").astype("float32")
            out[f"p_site_stage1_x_{c}"] = (p * x).astype("float32")
            added += 1
    print(f"✅ Added Stage-1 score interaction features: {added}")
    return out


def apply_high_confidence_filter(df, sample_weight_col="sample_weight", min_sample_weight=0.0, high_conf_quantile=0.0):
    """
    Optional high-confidence subset filtering.
    Use this only when reporting high-confidence subset evaluation, not as the default full-data result.
    """
    out = df.copy()
    before = len(out)
    if sample_weight_col in out.columns:
        w = pd.to_numeric(out[sample_weight_col], errors="coerce")
        keep = pd.Series(True, index=out.index)
        if min_sample_weight and min_sample_weight > 0:
            keep &= w >= float(min_sample_weight)
        if high_conf_quantile and high_conf_quantile > 0:
            q = float(w.quantile(float(high_conf_quantile)))
            keep &= w >= q
            print(f"High-confidence quantile threshold for {sample_weight_col}: q={high_conf_quantile} -> {q:.4f}")
        out = out.loc[keep].copy()
        print(f"✅ High-confidence filter by {sample_weight_col}: rows {before:,}->{len(out):,}")
    elif min_sample_weight > 0 or high_conf_quantile > 0:
        print(f"⚠️ High-confidence filter requested, but column not found: {sample_weight_col}; skipped")
    return out


def train_xgb(Xtr, ytr, Xva, yva, seed, params, wtr=None):
    if xgb is None:
        raise RuntimeError("pip install xgboost")
    clean = dict(params)
    clean.setdefault("objective", "binary:logistic")
    clean.setdefault("tree_method", "hist")
    clean.setdefault("n_jobs", -1)
    clean.setdefault("random_state", seed)
    clean.setdefault("verbosity", 0)
    clf = xgb.XGBClassifier(**clean)
    fit_kwargs = {}
    if wtr is not None:
        fit_kwargs["sample_weight"] = wtr
    clf.fit(Xtr, ytr, **fit_kwargs)
    return clf


def train_lgb_booster(Xtr, ytr, Xva, yva, seed, params, early_stop, wtr=None, wva=None):
    if lgb is None:
        raise RuntimeError("pip install lightgbm")
    lgb_params = dict(objective="binary", metric=["auc", "average_precision"], learning_rate=float(params["learning_rate"]), num_leaves=int(params["num_leaves"]), min_data_in_leaf=int(params["min_data_in_leaf"]), feature_fraction=float(params["feature_fraction"]), bagging_fraction=float(params["bagging_fraction"]), bagging_freq=int(params["bagging_freq"]), lambda_l2=float(params.get("lambda_l2", 1.0)), seed=int(seed), verbosity=-1, scale_pos_weight=float(params["scale_pos_weight"]))
    dtr = lgb.Dataset(Xtr, label=ytr, weight=wtr, free_raw_data=True)
    dva = lgb.Dataset(Xva, label=yva, weight=wva, reference=dtr, free_raw_data=True)
    booster = lgb.train(lgb_params, dtr, num_boost_round=int(params["n_estimators"]), valid_sets=[dva], valid_names=["val"], callbacks=[lgb.early_stopping(stopping_rounds=int(early_stop), verbose=False), lgb.log_evaluation(period=0)])
    return booster


def train_cat(Xtr, ytr, Xva, yva, seed, params, wtr=None, wva=None):
    if CatBoostClassifier is None:
        raise RuntimeError("pip install catboost")
    clf = CatBoostClassifier(**params, random_seed=seed, verbose=False)
    fit_kwargs = {"eval_set": (Xva, yva), "use_best_model": True}
    if wtr is not None:
        fit_kwargs["sample_weight"] = wtr
    clf.fit(Xtr, ytr, **fit_kwargs)
    return clf


class ResBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.BatchNorm1d(d), nn.Dropout(dropout), nn.Linear(d, d), nn.ReLU(), nn.BatchNorm1d(d))
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


def train_mlp(Xtr, ytr, Xva, yva, seed, params, wtr=None, wva=None):
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
    if model_name == "lgb":
        return model.predict(X, num_iteration=model.best_iteration)
    if model_name == "mlp":
        device = next(model.parameters()).device
        Xs = model.scaler_.transform(X).astype(np.float32)
        outs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(Xs), 4096):
                xb = torch.from_numpy(Xs[i:i+4096]).to(device)
                p = torch.sigmoid(model(xb)).cpu().numpy()
                outs.append(p)
        return np.concatenate(outs)
    return model.predict_proba(X)[:, 1]


def save_model(model_name, model, path):
    if model_name == "xgb":
        model.save_model(path)
    elif model_name == "lgb":
        model.save_model(path)
    elif model_name == "cat":
        model.save_model(path)
    elif model_name == "mlp":
        torch.save({"state_dict": model.state_dict(), "scaler_mean": model.scaler_.mean_, "scaler_scale": model.scaler_.scale_}, path)


def build_X_y(df, cell_type_col="cell_type", sample_weight_col="sample_weight", add_cell_type_onehot=True):
    y = pd.to_numeric(df["label_cell"], errors="coerce")
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int).to_numpy()
    if sample_weight_col in df.columns:
        w = pd.to_numeric(df[sample_weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        w = np.where(np.isfinite(w), w, 1.0)
        w = np.clip(w, 1e-6, None)
    else:
        w = np.ones(len(df), dtype=float)
    drop = {"chrom", "pos_based", "label_cell", "gene_id", "gene_name", sample_weight_col}
    if not add_cell_type_onehot:
        drop.add(cell_type_col)
    X = df.drop(columns=[c for c in df.columns if c in drop], errors="ignore").copy()
    if add_cell_type_onehot and cell_type_col in df.columns:
        ct = pd.get_dummies(df[cell_type_col].astype(str), prefix="ct", dtype=np.uint8)
        X = pd.concat([X, ct], axis=1)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    bad_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if bad_cols:
        X = X.drop(columns=bad_cols)
    X = downcast_df_numeric(X)
    return df, X, y, w, list(X.columns)


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


def train_one_split(df, X, y, w, feature_cols, out_dir, args, seed, seed_offset=0, cell_type_col="cell_type"):
    ensure_dir(out_dir)
    groups = make_groups(df, block_bp=args.block_bp)
    idx = np.arange(len(df))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed + seed_offset)
    trainval_idx, test_idx = next(gss1.split(idx, y, groups))
    groups_tv = groups[trainval_idx]
    y_tv = y[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_size / (1 - args.test_size), random_state=seed + seed_offset + 1)
    tr_rel, va_rel = next(gss2.split(trainval_idx, y_tv, groups_tv))
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]
    if args.max_train_rows > 0 and len(train_idx) > args.max_train_rows:
        rng = np.random.RandomState(seed + 999 + seed_offset)
        train_idx = rng.choice(train_idx, size=args.max_train_rows, replace=False)
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X.iloc[train_idx])
    Xva = imp.transform(X.iloc[val_idx])
    Xte = imp.transform(X.iloc[test_idx])
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]
    wtr, wva, wte = w[train_idx], w[val_idx], w[test_idx]
    if getattr(args, "sample_weight_power", 1.0) != 1.0:
        powv = float(args.sample_weight_power)
        wtr = np.power(wtr, powv)
        wva = np.power(wva, powv)
        wte = np.power(wte, powv)
    spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
    meta_test = df.iloc[test_idx][["chrom", "pos_based", cell_type_col]].reset_index(drop=True)
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    rows = []
    val_preds, test_preds = {}, {}
    print(f"  🧪 sample_weight | train min/median/max = {wtr.min():.4f}/{np.median(wtr):.4f}/{wtr.max():.4f}")
    for m in [x for x in models if x != "blend"]:
        mdir = os.path.join(out_dir, m)
        ensure_dir(mdir)
        print(f"  ▶ Training {m.upper()} | train={len(ytr):,} val={len(yva):,} test={len(yte):,}")
        if m == "xgb":
            params = dict(n_estimators=args.xgb_estimators, learning_rate=args.xgb_lr, max_depth=args.xgb_depth, subsample=args.xgb_subsample, colsample_bytree=args.xgb_colsample, reg_lambda=args.xgb_lambda, reg_alpha=args.xgb_alpha, gamma=args.xgb_gamma, min_child_weight=args.xgb_min_child_weight, scale_pos_weight=spw, tree_method=args.xgb_tree_method)
            model = train_xgb(Xtr, ytr, Xva, yva, seed + 10 + seed_offset, params, wtr=wtr)
            model_path = os.path.join(mdir, "model.xgb.json")
        elif m == "lgb":
            params = dict(n_estimators=args.lgb_estimators, learning_rate=args.lgb_lr, num_leaves=args.lgb_num_leaves, min_data_in_leaf=args.lgb_min_data_in_leaf, feature_fraction=args.lgb_feature_fraction, bagging_fraction=args.lgb_bagging_fraction, bagging_freq=args.lgb_bagging_freq, scale_pos_weight=spw, lambda_l2=1.0)
            model = train_lgb_booster(Xtr, ytr, Xva, yva, seed + 20 + seed_offset, params, early_stop=args.lgb_early_stop, wtr=wtr, wva=wva)
            model_path = os.path.join(mdir, "model.lgb.txt")
        elif m == "cat":
            params = dict(iterations=args.cat_iters, learning_rate=args.cat_lr, depth=args.cat_depth, l2_leaf_reg=args.cat_l2, loss_function="Logloss", eval_metric="AUC", class_weights=[1.0, float(spw)], od_type="Iter", od_wait=args.cat_early_stop)
            model = train_cat(Xtr, ytr, Xva, yva, seed + 30 + seed_offset, params, wtr=wtr, wva=wva)
            model_path = os.path.join(mdir, "model.cat.cbm")
        elif m == "mlp":
            params = dict(hidden_dim=args.mlp_hidden_dim, depth=args.mlp_depth, dropout=args.mlp_dropout, lr=args.mlp_lr, weight_decay=args.mlp_weight_decay, batch_size=args.mlp_batch_size, epochs=args.mlp_epochs, patience=args.mlp_patience)
            model = train_mlp(Xtr, ytr, Xva, yva, seed + 40 + seed_offset, params, wtr=wtr, wva=wva)
            model_path = os.path.join(mdir, "model.mlp.pt")
        else:
            continue
        pva = predict(m, model, Xva)
        pte = predict(m, model, Xte)
        val_preds[m], test_preds[m] = pva, pte
        va_auc, va_ap = safe_auc(yva, pva), safe_ap(yva, pva)
        te_auc, te_ap = safe_auc(yte, pte), safe_ap(yte, pte)
        te_top = topk_report(yte, pte)
        best_t, _ = best_f1_threshold(yva, pva)
        save_pr(yte, pte, os.path.join(mdir, "pr_curve.csv"))
        save_fold_predictions(meta_test, yte, pte, os.path.join(mdir, "test_predictions.csv"))
        save_model(m, model, model_path)
        feature_importance(m, model, feature_cols, os.path.join(mdir, "feature_importance.csv"))
        rows.append({"seed": seed, "model": m, "val_auc": va_auc, "val_ap": va_ap, "test_ap": te_ap, "test_auc": te_auc, "test_top5_R": te_top["top5%_R"], "test_f1_bestT": float(f1_score(yte, (pte >= best_t).astype(int), zero_division=0)), "test_weight_mean": float(np.mean(wte))})
        print(f"    ✅ {m.upper()} done | val_auc={va_auc:.4f} test_auc={te_auc:.4f} test_ap={te_ap:.4f}")
    if "blend" in models and len(val_preds) >= 2:
        weights, blend_df = blend_weights_from_val(yva, val_preds)
        pva = sum(weights[m] * val_preds[m] for m in weights)
        pte = sum(weights[m] * test_preds[m] for m in weights)
        va_auc, va_ap = safe_auc(yva, pva), safe_ap(yva, pva)
        te_auc, te_ap = safe_auc(yte, pte), safe_ap(yte, pte)
        te_top = topk_report(yte, pte)
        best_t, _ = best_f1_threshold(yva, pva)
        mdir = os.path.join(out_dir, "blend")
        ensure_dir(mdir)
        save_pr(yte, pte, os.path.join(mdir, "pr_curve.csv"))
        save_fold_predictions(meta_test, yte, pte, os.path.join(mdir, "test_predictions.csv"))
        blend_df["weight"] = blend_df["model"].map(weights)
        blend_df.to_csv(os.path.join(mdir, "blend_weights.csv"), index=False)
        rows.append({"seed": seed, "model": "blend", "val_auc": va_auc, "val_ap": va_ap, "test_ap": te_ap, "test_auc": te_auc, "test_top5_R": te_top["top5%_R"], "test_f1_bestT": float(f1_score(yte, (pte >= best_t).astype(int), zero_division=0)), "test_weight_mean": float(np.mean(wte))})
        print(f"    ✅ BLEND done | val_auc={va_auc:.4f} test_auc={te_auc:.4f} test_ap={te_ap:.4f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=r"model_input/Brain_CPG_ALL_UNION/stage2_cell_static_dynamic_union.csv")
    ap.add_argument("--out_dir", default=r"model_output/stage2_union_v2")
    ap.add_argument("--models", default="xgb,lgb,cat,mlp,blend")
    ap.add_argument("--block_bp", type=int, default=10000)
    ap.add_argument("--val_size", type=float, default=0.20)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_list", default="")
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--max_train_rows", type=int, default=0)
    ap.add_argument("--locto", action="store_true")
    ap.add_argument("--cell_type_col", default="cell_type")
    ap.add_argument("--holdout_cell_types", default="")
    ap.add_argument("--holdout_groups", default="")
    ap.add_argument("--sample_weight_col", default="sample_weight")
    ap.add_argument("--disable_cell_type_onehot", action="store_true")
    ap.add_argument("--stage1_scores_csv", default="")
    ap.add_argument("--stage1_score_col", default="p_site")
    ap.add_argument("--candidate_top_frac", type=float, default=1.0)
    ap.add_argument("--use_stage1_soft_feature", action="store_true")
    ap.add_argument("--add_stage1_score_transforms", action="store_true",
                    help="Add logit/rank/centered/square transforms of p_site_stage1")
    ap.add_argument("--add_stage1_score_interactions", action="store_true",
                    help="Add interactions between p_site_stage1 and 5mC-derived Stage-2 features")
    ap.add_argument("--min_sample_weight", type=float, default=0.0,
                    help="Optional high-confidence subset filter: keep rows with sample_weight >= threshold")
    ap.add_argument("--high_conf_quantile", type=float, default=0.0,
                    help="Optional high-confidence subset filter: keep rows with sample_weight >= this quantile, e.g. 0.5")
    ap.add_argument("--sample_weight_power", type=float, default=1.0,
                    help="Raise sample_weight to this power during training; >1 emphasizes high-confidence rows")
    # xgb
    ap.add_argument("--xgb_tree_method", default="hist")
    ap.add_argument("--xgb_estimators", type=int, default=1000)
    ap.add_argument("--xgb_lr", type=float, default=0.03)
    ap.add_argument("--xgb_depth", type=int, default=5)
    ap.add_argument("--xgb_subsample", type=float, default=0.8)
    ap.add_argument("--xgb_colsample", type=float, default=0.7)
    ap.add_argument("--xgb_lambda", type=float, default=2.0)
    ap.add_argument("--xgb_alpha", type=float, default=0.1)
    ap.add_argument("--xgb_gamma", type=float, default=0.5)
    ap.add_argument("--xgb_min_child_weight", type=float, default=6.0)
    # lgb
    ap.add_argument("--lgb_estimators", type=int, default=6000)
    ap.add_argument("--lgb_lr", type=float, default=0.03)
    ap.add_argument("--lgb_num_leaves", type=int, default=63)
    ap.add_argument("--lgb_min_data_in_leaf", type=int, default=40)
    ap.add_argument("--lgb_feature_fraction", type=float, default=0.8)
    ap.add_argument("--lgb_bagging_fraction", type=float, default=0.8)
    ap.add_argument("--lgb_bagging_freq", type=int, default=1)
    ap.add_argument("--lgb_early_stop", type=int, default=200)
    # cat
    ap.add_argument("--cat_iters", type=int, default=3000)
    ap.add_argument("--cat_lr", type=float, default=0.03)
    ap.add_argument("--cat_depth", type=int, default=8)
    ap.add_argument("--cat_l2", type=float, default=6.0)
    ap.add_argument("--cat_early_stop", type=int, default=150)
    # mlp
    ap.add_argument("--mlp_hidden_dim", type=int, default=256)
    ap.add_argument("--mlp_depth", type=int, default=3)
    ap.add_argument("--mlp_dropout", type=float, default=0.15)
    ap.add_argument("--mlp_lr", type=float, default=1e-3)
    ap.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    ap.add_argument("--mlp_batch_size", type=int, default=1024)
    ap.add_argument("--mlp_epochs", type=int, default=100)
    ap.add_argument("--mlp_patience", type=int, default=10)
    args = ap.parse_args()
    ensure_dir(args.out_dir)
    print("=" * 100)
    print("📥 Loading Stage-2 dataset...")
    df = pd.read_csv(args.in_csv, low_memory=False)
    need = ["chrom", "pos_based", args.cell_type_col, "label_cell"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["pos_based"] = pd.to_numeric(df["pos_based"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["pos_based"]).copy()
    df["pos_based"] = df["pos_based"].astype(int)
    if args.stage1_scores_csv.strip():
        print("📎 Loading Stage-1 scores...")
        scores = load_stage1_scores(args.stage1_scores_csv, score_col=args.stage1_score_col)
        report_stage1_score_coverage(df, scores, score_col=args.stage1_score_col)
        if args.candidate_top_frac < 1.0:
            before_rows = len(df)
            before_sites = df[["chrom", "pos_based"]].drop_duplicates().shape[0]
            df, thr, kept_sites, out_sites = apply_candidate_gating(df, scores, args.candidate_top_frac, score_col=args.stage1_score_col)
            print(f"✅ Hard gating: kept top {args.candidate_top_frac:.2f} global Stage1 sites | threshold≈{thr:.4f} | rows {before_rows:,}->{len(df):,} | sites {before_sites:,}->{out_sites:,}")
        if args.use_stage1_soft_feature:
            df = add_stage1_soft_feature(df, scores, score_col=args.stage1_score_col)
            cov = float(df["p_site_stage1"].notna().mean())
            print(f"✅ Added soft Stage1 feature: p_site_stage1 | row coverage={cov:.2%}")
            if args.add_stage1_score_transforms:
                df = add_stage1_score_transforms(df)
                print("✅ Added Stage-1 score transforms")
            if args.add_stage1_score_interactions:
                df = add_stage1_score_interactions(df)
    df = apply_high_confidence_filter(
        df,
        sample_weight_col=args.sample_weight_col,
        min_sample_weight=args.min_sample_weight,
        high_conf_quantile=args.high_conf_quantile,
    )
    if args.max_rows > 0 and len(df) > args.max_rows:
        rng = np.random.RandomState(args.seed)
        keep_idx = rng.choice(np.arange(len(df)), size=args.max_rows, replace=False)
        df = df.iloc[keep_idx].reset_index(drop=True)
        print(f"⚡ max_rows active: reduced to {len(df):,} rows")
    df = downcast_df_numeric(df)
    df, X, y, w, feature_cols = build_X_y(df, cell_type_col=args.cell_type_col, sample_weight_col=args.sample_weight_col, add_cell_type_onehot=(not args.disable_cell_type_onehot))
    print("=" * 100)
    print("🌲 Stage-2 v2 (cell) training")
    print("=" * 100)
    print(f"Input: {args.in_csv}")
    print(f"N={len(df):,} features={len(feature_cols)} pos={int(y.sum()):,} ({y.mean():.2%})")
    print(f"Models: {args.models}")
    print(f"LOCTO: {args.locto}")
    print(f"sample_weight summary: min={w.min():.4f} median={np.median(w):.4f} max={w.max():.4f}")
    print(f"cell_type one-hot: {not args.disable_cell_type_onehot}")
    print(f"Stage1 score transforms: {args.add_stage1_score_transforms} | interactions: {args.add_stage1_score_interactions}")
    print(f"high-confidence filter: min_sample_weight={args.min_sample_weight} | high_conf_quantile={args.high_conf_quantile} | sample_weight_power={args.sample_weight_power}")
    print("=" * 100)
    if args.seed_list.strip():
        seed_list = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]
    else:
        seed_list = [args.seed]
    if args.locto:
        raise NotImplementedError("v2 keeps focus on non-LOCTO full evaluation; LOCTO can be ported if needed.")
    all_rows = []
    for i, seed in enumerate(seed_list):
        print(f"\n===== Seed {seed} ({i+1}/{len(seed_list)}) =====")
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        out = train_one_split(df, X, y, w, feature_cols, seed_dir, args, seed=seed, seed_offset=0, cell_type_col=args.cell_type_col)
        all_rows.append(out)
    all_df = pd.concat(all_rows, axis=0, ignore_index=True)
    all_df.to_csv(os.path.join(args.out_dir, "summary_all_seeds.csv"), index=False)
    agg = all_df.groupby("model")[["val_auc", "val_ap", "test_auc", "test_ap", "test_top5_R", "test_f1_bestT"]].agg(["mean", "std", "min", "max"])
    agg.to_csv(os.path.join(args.out_dir, "summary_agg.csv"))
    print("\n✅ Finished non-LOCTO")
    print(agg.to_string())


if __name__ == "__main__":
    main()
