#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P6_train_deep5hmc_like_comparator_v2.py

A more stable and fair Deep5hmC-inspired regional comparator for the user's current data.

Key upgrades over v1
--------------------
1) FIXED SPLIT across seeds
   - `split_seed` controls train/val/test split once.
   - `seed_list` controls model initialization / optimization randomness only.
   - This separates split variance from model variance.

2) REGION-GROUPED SPLIT
   - Split groups are defined by `chrom|window_start`, so the same genomic window
     across different cell types stays in the same split.
   - This is more appropriate for region-level evaluation than chromosome-only grouping.

3) BETTER REPRODUCIBILITY
   - Sets NumPy / PyTorch random seeds per run.
   - Saves split membership for auditability.

4) ROBUSTER TRAINING DEFAULTS
   - Uses SmoothL1Loss (Huber-style) by default.
   - Slightly stronger regularization and dropout.

Positioning
-----------
- This is still NOT an exact reproduction of the original Deep5hmC architecture.
- It is an adapted, runnable regional comparator inspired by Deep5hmC for the user's
  current tabular data format.
- Recommended use: external comparator / supplementary analysis.
"""

import os
import json
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:
    raise RuntimeError("PyTorch is required for this script. Please install torch.") from e


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


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


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
# data prep
# ---------------------------------------------------------------------

def load_stage2_table(stage2_csv: str) -> pd.DataFrame:
    df = pd.read_csv(stage2_csv, low_memory=False)
    need = ["chrom", "pos_based", "cell_type"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"stage2_csv missing required columns: {miss}")

    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["pos_based"] = safe_numeric(df["pos_based"])
    df = df[df["pos_based"].notna()].copy()
    df["pos_based"] = df["pos_based"].astype(int)
    df["cell_type"] = df["cell_type"].astype(str)
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
    df[truth_signal_col] = safe_numeric(df[truth_signal_col])

    if weight_col:
        if weight_col not in df.columns:
            raise ValueError(f"weight_col={weight_col} not in truth_csv")
        df["row_weight"] = safe_numeric(df[weight_col]).fillna(1.0).astype(float)
    else:
        df["row_weight"] = 1.0

    return df[["chrom", "pos_based", "cell_type", truth_signal_col, "row_weight"]].copy()


def split_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Heuristic split:
    - epigenetic/dynamic-like: mC_ / cell_hmc_ / cell_mc_ / promoter-related / site-related
    - static/sequence-like: remaining numeric cols excluding ids/labels/weights
    """
    exclude = {
        "chrom", "pos_based", "cell_type",
        "label_cell", "label_site", "sample_weight",
        "gene_id", "gene_name",
    }

    epi_patterns = [
        "mC_", "cell_hmc_", "cell_mc_",
        "promoter", "site_", "_z", "_delta", "_rank", "_over_"
    ]

    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    epi_cols = []
    static_cols = []
    for c in numeric_cols:
        is_epi = any(p in c for p in epi_patterns)
        if is_epi:
            epi_cols.append(c)
        else:
            static_cols.append(c)

    if len(static_cols) == 0:
        static_cols = numeric_cols.copy()
    if len(epi_cols) == 0:
        epi_cols = []

    return static_cols, epi_cols


def build_window_dataset(
    stage2_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    truth_signal_col: str,
    window_bp: int,
    min_cpgs: int,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = stage2_df.merge(
        truth_df,
        on=["chrom", "pos_based", "cell_type"],
        how="inner"
    )
    if len(df) == 0:
        raise RuntimeError("No merged rows between stage2_csv and truth_csv.")

    static_cols, epi_cols = split_feature_groups(df)

    df["window_start"] = ((df["pos_based"] - 1) // int(window_bp)) * int(window_bp) + 1
    df["window_end"] = df["window_start"] + int(window_bp) - 1

    rows = []
    group_keys = ["chrom", "window_start", "window_end", "cell_type"]

    for keys, sub in df.groupby(group_keys, sort=False):
        chrom, ws, we, ct = keys
        n_cpgs = int(len(sub))
        if n_cpgs < int(min_cpgs):
            continue

        w = sub["row_weight"].astype(float).to_numpy()
        y = safe_numeric(sub[truth_signal_col]).to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
        if mask.sum() == 0:
            continue

        # region continuous target: log1p weighted sum
        y_sum = float(np.sum(y[mask] * w[mask]))
        y_reg = np.log1p(max(y_sum, 0.0))

        row = {
            "chrom": chrom,
            "window_start": int(ws),
            "window_end": int(we),
            "cell_type": ct,
            "n_cpgs": n_cpgs,
            "target_y": y_reg,
            # split group: same genomic window across cell types stays together
            "split_group": f"{chrom}|{int(ws)}",
        }

        for c in static_cols:
            vals = safe_numeric(sub[c]).to_numpy(dtype=float)
            row[c] = np.nanmean(vals)

        for c in epi_cols:
            vals = safe_numeric(sub[c]).to_numpy(dtype=float)
            row[c] = np.nanmean(vals)

        rows.append(row)

    out = pd.DataFrame(rows)
    return out, static_cols, epi_cols


# ---------------------------------------------------------------------
# model
# ---------------------------------------------------------------------

class BranchMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Deep5hmCLikeRegressor(nn.Module):
    def __init__(self, static_dim: int, epi_dim: int, n_cell_types: int, hidden: int = 96, dropout: float = 0.30):
        super().__init__()

        self.static_branch = BranchMLP(static_dim, hidden=hidden, dropout=dropout)

        if epi_dim > 0:
            self.epi_branch = BranchMLP(epi_dim, hidden=hidden, dropout=dropout)
            fusion_in = hidden * 2 + n_cell_types + 1  # + n_cpgs
        else:
            self.epi_branch = None
            fusion_in = hidden + n_cell_types + 1

        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_static, x_epi, x_ct, x_n):
        h1 = self.static_branch(x_static)
        feats = [h1]

        if self.epi_branch is not None and x_epi is not None:
            h2 = self.epi_branch(x_epi)
            feats.append(h2)

        feats.extend([x_ct, x_n])
        h = torch.cat(feats, dim=1)
        out = self.head(h).squeeze(1)
        return out


# ---------------------------------------------------------------------
# prep and splitting
# ---------------------------------------------------------------------

def prepare_X(win_df: pd.DataFrame, static_cols: List[str], epi_cols: List[str]):
    ct = pd.get_dummies(win_df["cell_type"].astype(str), prefix="ct", dtype=np.float32)

    Xs = win_df[static_cols].copy()
    Xe = win_df[epi_cols].copy() if len(epi_cols) > 0 else pd.DataFrame(index=win_df.index)
    Xn = win_df[["n_cpgs"]].astype(float).copy()
    y = win_df["target_y"].astype(float).to_numpy(dtype=np.float32)
    groups = win_df["split_group"].astype(str).to_numpy()

    for c in Xs.columns:
        Xs[c] = safe_numeric(Xs[c])
    for c in Xe.columns:
        Xe[c] = safe_numeric(Xe[c])

    return Xs, Xe, ct, Xn, y, groups


def make_fixed_split(win_df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, split_seed: int, test_size: float, val_size: float):
    idx = np.arange(len(win_df))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=split_seed)
    trva_idx, te_idx = next(gss1.split(idx, y, groups))

    y_trva = y[trva_idx]
    g_trva = groups[trva_idx]
    val_frac_in_trva = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_in_trva, random_state=split_seed + 1)
    tr_rel, va_rel = next(gss2.split(trva_idx, y_trva, g_trva))
    tr_idx = trva_idx[tr_rel]
    va_idx = trva_idx[va_rel]

    return tr_idx, va_idx, te_idx


def fit_scalers_imputers(Xtr_s, Xtr_e, Xtr_ct, Xtr_n):
    imp_s = SimpleImputer(strategy="median")
    sc_s = StandardScaler()
    Xtr_s2 = sc_s.fit_transform(imp_s.fit_transform(Xtr_s)).astype(np.float32)

    if Xtr_e.shape[1] > 0:
        imp_e = SimpleImputer(strategy="median")
        sc_e = StandardScaler()
        Xtr_e2 = sc_e.fit_transform(imp_e.fit_transform(Xtr_e)).astype(np.float32)
    else:
        imp_e, sc_e = None, None
        Xtr_e2 = None

    Xtr_ct2 = Xtr_ct.to_numpy(dtype=np.float32)

    imp_n = SimpleImputer(strategy="median")
    sc_n = StandardScaler()
    Xtr_n2 = sc_n.fit_transform(imp_n.fit_transform(Xtr_n)).astype(np.float32)

    return (imp_s, sc_s, imp_e, sc_e, imp_n, sc_n), (Xtr_s2, Xtr_e2, Xtr_ct2, Xtr_n2)


def transform_with_preprocessors(Xs, Xe, Xct, Xn, preprocessors):
    imp_s, sc_s, imp_e, sc_e, imp_n, sc_n = preprocessors

    Xs2 = sc_s.transform(imp_s.transform(Xs)).astype(np.float32)
    if Xe.shape[1] > 0:
        Xe2 = sc_e.transform(imp_e.transform(Xe)).astype(np.float32)
    else:
        Xe2 = None

    Xct2 = Xct.to_numpy(dtype=np.float32)
    Xn2 = sc_n.transform(imp_n.transform(Xn)).astype(np.float32)
    return Xs2, Xe2, Xct2, Xn2


def make_loader(Xs, Xe, Xct, Xn, y, batch_size, shuffle):
    tensors = [torch.from_numpy(Xs)]
    if Xe is not None:
        tensors.append(torch.from_numpy(Xe))
    else:
        tensors.append(torch.zeros((Xs.shape[0], 0), dtype=torch.float32))
    tensors.append(torch.from_numpy(Xct))
    tensors.append(torch.from_numpy(Xn))
    tensors.append(torch.from_numpy(y.astype(np.float32)))
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ---------------------------------------------------------------------
# train/eval
# ---------------------------------------------------------------------

def run_one_seed(
    win_df,
    static_cols,
    epi_cols,
    tr_idx,
    va_idx,
    te_idx,
    seed,
    epochs,
    batch_size,
    lr,
    weight_decay,
    patience,
):
    set_global_seed(seed)

    Xs, Xe, Xct, Xn, y, _ = prepare_X(win_df, static_cols, epi_cols)

    Xtr_s, Xva_s, Xte_s = Xs.iloc[tr_idx], Xs.iloc[va_idx], Xs.iloc[te_idx]
    Xtr_e, Xva_e, Xte_e = Xe.iloc[tr_idx], Xe.iloc[va_idx], Xe.iloc[te_idx]
    Xtr_ct, Xva_ct, Xte_ct = Xct.iloc[tr_idx], Xct.iloc[va_idx], Xct.iloc[te_idx]
    Xtr_n, Xva_n, Xte_n = Xn.iloc[tr_idx], Xn.iloc[va_idx], Xn.iloc[te_idx]

    ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

    preprocessors, (Xtr_s2, Xtr_e2, Xtr_ct2, Xtr_n2) = fit_scalers_imputers(Xtr_s, Xtr_e, Xtr_ct, Xtr_n)
    Xva_s2, Xva_e2, Xva_ct2, Xva_n2 = transform_with_preprocessors(Xva_s, Xva_e, Xva_ct, Xva_n, preprocessors)
    Xte_s2, Xte_e2, Xte_ct2, Xte_n2 = transform_with_preprocessors(Xte_s, Xte_e, Xte_ct, Xte_n, preprocessors)

    dl_tr = make_loader(Xtr_s2, Xtr_e2, Xtr_ct2, Xtr_n2, ytr, batch_size=batch_size, shuffle=True)
    dl_va = make_loader(Xva_s2, Xva_e2, Xva_ct2, Xva_n2, yva, batch_size=max(batch_size, 1024), shuffle=False)
    dl_te = make_loader(Xte_s2, Xte_e2, Xte_ct2, Xte_n2, yte, batch_size=max(batch_size, 1024), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Deep5hmCLikeRegressor(
        static_dim=Xtr_s2.shape[1],
        epi_dim=0 if Xtr_e2 is None else Xtr_e2.shape[1],
        n_cell_types=Xtr_ct2.shape[1],
        hidden=96,
        dropout=0.30,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(beta=0.5)

    best_state = None
    best_val_s = -1e18
    bad = 0

    for ep in range(epochs):
        model.train()
        for xb_s, xb_e, xb_ct, xb_n, yb in dl_tr:
            xb_s = xb_s.to(device)
            xb_e = xb_e.to(device)
            xb_ct = xb_ct.to(device)
            xb_n = xb_n.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb_s, xb_e if xb_e.shape[1] > 0 else None, xb_ct, xb_n)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        pva = []
        yva_collect = []
        with torch.no_grad():
            for xb_s, xb_e, xb_ct, xb_n, yb in dl_va:
                xb_s = xb_s.to(device)
                xb_e = xb_e.to(device)
                xb_ct = xb_ct.to(device)
                xb_n = xb_n.to(device)
                pred = model(xb_s, xb_e if xb_e.shape[1] > 0 else None, xb_ct, xb_n)
                pva.append(pred.cpu().numpy())
                yva_collect.append(yb.numpy())
        pva = np.concatenate(pva)
        yva_np = np.concatenate(yva_collect)
        val_s = safe_spearman(yva_np, pva)

        if np.isfinite(val_s) and val_s > best_val_s:
            best_val_s = val_s
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    pte = []
    yte_collect = []
    with torch.no_grad():
        for xb_s, xb_e, xb_ct, xb_n, yb in dl_te:
            xb_s = xb_s.to(device)
            xb_e = xb_e.to(device)
            xb_ct = xb_ct.to(device)
            xb_n = xb_n.to(device)
            pred = model(xb_s, xb_e if xb_e.shape[1] > 0 else None, xb_ct, xb_n)
            pte.append(pred.cpu().numpy())
            yte_collect.append(yb.numpy())
    pte = np.concatenate(pte)
    yte_np = np.concatenate(yte_collect)

    test_meta = win_df.iloc[te_idx][["cell_type"]].reset_index(drop=True)
    per_group_rows = []
    for grp, sub_idx in test_meta.groupby("cell_type").groups.items():
        yt = yte_np[list(sub_idx)]
        yp = pte[list(sub_idx)]
        per_group_rows.append({
            "cell_type": grp,
            "n_windows": int(len(sub_idx)),
            "spearman": safe_spearman(yt, yp),
            "pearson": safe_pearson(yt, yp),
            "rmse": safe_rmse(yt, yp),
            "mae": safe_mae(yt, yp),
        })
    per_group_df = pd.DataFrame(per_group_rows)

    out = {
        "seed": seed,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "val_spearman_best": best_val_s,
        "test_spearman": safe_spearman(yte_np, pte),
        "test_pearson": safe_pearson(yte_np, pte),
        "test_rmse": safe_rmse(yte_np, pte),
        "test_mae": safe_mae(yte_np, pte),
        "macro_mean_spearman": float(per_group_df["spearman"].mean()) if len(per_group_df) > 0 else np.nan,
        "macro_median_spearman": float(per_group_df["spearman"].median()) if len(per_group_df) > 0 else np.nan,
        "macro_mean_pearson": float(per_group_df["pearson"].mean()) if len(per_group_df) > 0 else np.nan,
        "macro_median_pearson": float(per_group_df["pearson"].median()) if len(per_group_df) > 0 else np.nan,
        "n_groups": int(per_group_df["cell_type"].nunique()) if len(per_group_df) > 0 else 0,
    }

    win_pred_df = win_df.iloc[te_idx][[
        "chrom", "window_start", "window_end", "cell_type", "n_cpgs", "target_y"
    ]].reset_index(drop=True).copy()

    win_pred_df = win_pred_df.rename(columns={"target_y": "truth_window"})
    win_pred_df["pred_window"] = pte
    win_pred_df["seed"] = seed

    return out, per_group_df, win_pred_df


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_csv", required=True)
    ap.add_argument("--truth_csv", required=True)
    ap.add_argument("--truth_signal_col", default="hmc_signal_cont")
    ap.add_argument("--weight_col", default="sample_weight")
    ap.add_argument("--window_bp", type=int, default=1000)
    ap.add_argument("--min_cpgs", type=int, default=3)
    ap.add_argument("--seed_list", default="42,52,62,72,82")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    seeds = parse_seed_list(args.seed_list)

    stage2_df = load_stage2_table(args.stage2_csv)
    truth_df = load_truth_long(args.truth_csv, args.truth_signal_col, args.weight_col)
    win_df, static_cols, epi_cols = build_window_dataset(
        stage2_df=stage2_df,
        truth_df=truth_df,
        truth_signal_col=args.truth_signal_col,
        window_bp=args.window_bp,
        min_cpgs=args.min_cpgs,
    )

    Xs, Xe, Xct, Xn, y, groups = prepare_X(win_df, static_cols, epi_cols)
    tr_idx, va_idx, te_idx = make_fixed_split(
        win_df=win_df,
        y=y,
        groups=groups,
        split_seed=args.split_seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    split_df = pd.DataFrame({
        "row_idx": np.concatenate([tr_idx, va_idx, te_idx]),
        "split": (["train"] * len(tr_idx)) + (["val"] * len(va_idx)) + (["test"] * len(te_idx)),
    }).sort_values("row_idx")
    split_df.to_csv(os.path.join(args.out_dir, "fixed_split_rows.csv"), index=False)

    print("=" * 100)
    print("Deep5hmC-inspired comparator v2")
    print("=" * 100)
    print(f"stage2_csv      : {args.stage2_csv}")
    print(f"truth_csv       : {args.truth_csv}")
    print(f"window_bp       : {args.window_bp}")
    print(f"min_cpgs        : {args.min_cpgs}")
    print(f"window_rows     : {len(win_df):,}")
    print(f"static_cols     : {len(static_cols)}")
    print(f"epi_cols        : {len(epi_cols)}")
    print(f"cell_types      : {sorted(win_df['cell_type'].unique().tolist())}")
    print(f"split_seed      : {args.split_seed}")
    print(f"n_train/val/test: {len(tr_idx):,} / {len(va_idx):,} / {len(te_idx):,}")
    print("=" * 100)

    all_rows = []
    all_group_rows = []
    all_window_rows = []

    for s in seeds:
        print(f"\n===== Model seed {s} =====")
        out, per_group_df, win_pred_df = run_one_seed(
            win_df=win_df,
            static_cols=static_cols,
            epi_cols=epi_cols,
            tr_idx=tr_idx,
            va_idx=va_idx,
            te_idx=te_idx,
            seed=s,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
        all_rows.append(out)

        tmp = per_group_df.copy()
        tmp["seed"] = s
        all_group_rows.append(tmp)

        print(
            f"test_spearman={out['test_spearman']:.4f} | "
            f"macro_mean_s={out['macro_mean_spearman']:.4f} | "
            f"macro_median_s={out['macro_median_spearman']:.4f}"
        )
        win_pred_df.to_csv(
            os.path.join(args.out_dir, f"window_predictions_seed_{s}.csv"),
            index=False
        )
        all_window_rows.append(win_pred_df)

    by_seed = pd.DataFrame(all_rows)
    by_seed.to_csv(os.path.join(args.out_dir, "summary_by_seed.csv"), index=False)

    metric_cols = [
        "test_spearman", "test_pearson", "test_rmse", "test_mae",
        "macro_mean_spearman", "macro_median_spearman",
        "macro_mean_pearson", "macro_median_pearson",
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
                "spearman_std": float(pd.to_numeric(sub["spearman"], errors="coerce").std(ddof=1)) if len(sub) >= 2 else np.nan,
                "pearson_mean": float(pd.to_numeric(sub["pearson"], errors="coerce").mean()),
                "pearson_std": float(pd.to_numeric(sub["pearson"], errors="coerce").std(ddof=1)) if len(sub) >= 2 else np.nan,
                "rmse_mean": float(pd.to_numeric(sub["rmse"], errors="coerce").mean()),
                "rmse_std": float(pd.to_numeric(sub["rmse"], errors="coerce").std(ddof=1)) if len(sub) >= 2 else np.nan,
                "mae_mean": float(pd.to_numeric(sub["mae"], errors="coerce").mean()),
                "mae_std": float(pd.to_numeric(sub["mae"], errors="coerce").std(ddof=1)) if len(sub) >= 2 else np.nan,
            })
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "summary_per_group_agg.csv"), index=False)

        if len(all_window_rows) > 0:
            all_win = pd.concat(all_window_rows, axis=0, ignore_index=True)
            all_win.to_csv(os.path.join(args.out_dir, "window_predictions_by_seed.csv"), index=False)

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

    with open(os.path.join(args.out_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("\n✅ Finished.")
    print("Saved:")
    print(" -", os.path.join(args.out_dir, "fixed_split_rows.csv"))
    print(" -", os.path.join(args.out_dir, "summary_by_seed.csv"))
    print(" -", os.path.join(args.out_dir, "summary_agg.csv"))
    print(" -", os.path.join(args.out_dir, "summary_per_group_by_seed.csv"))
    print(" -", os.path.join(args.out_dir, "summary_per_group_agg.csv"))


if __name__ == "__main__":
    main()
