#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P1_make_long_and_labels.py
Stage1: wide (union/intersection) -> long (site×cell) + label_cell + label_site.

【本版本已修改】
- 在计算完 site_role 后，直接删除 uncertain 位点
- 后续输出的 long/site/stage1 全部只保留 pos/neg
- 这样 P2/P3/P4/P5 默认就不会再处理 uncertain 数据
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Optional

# -------------------- GMM config --------------------
GMM_COMPONENTS = 2
GMM_RANDOM_STATE = 42


def fit_gmm_threshold(values: np.ndarray, min_samples: int, fallback: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size < int(min_samples):
        return float(fallback)

    X = x.reshape(-1, 1)
    try:
        gmm = GaussianMixture(n_components=GMM_COMPONENTS, random_state=GMM_RANDOM_STATE)
        gmm.fit(X)
        means = np.sort(gmm.means_.flatten())
        thr = float((means[0] + means[1]) / 2.0)
        if (not np.isfinite(thr)) or thr < 0:
            return float(fallback)
        return thr
    except Exception:
        return float(fallback)


def build_input_path(base_dir: str, input_mode: str, p0_high_conf_k: Optional[int]) -> str:
    m = input_mode.lower().strip()
    if m not in ("union", "intersection"):
        raise ValueError("--input_mode must be union or intersection")

    if p0_high_conf_k is None:
        return os.path.join(base_dir, f"ALL_cells_{m}_CpG_4cols_raw.csv")
    else:
        return os.path.join(base_dir, f"ALL_cells_{m}_CpG_4cols_high_conf_hmc{int(p0_high_conf_k)}.csv")


def default_outputs(base_dir: str, input_mode: str):
    m = input_mode.lower().strip()
    long_csv = os.path.join(base_dir, f"ALL_cells_{m}_CpG_long_table.csv")
    thr_csv = os.path.join(base_dir, f"ALL_cells_{m}_cell_thresholds.csv")
    site_csv = os.path.join(base_dir, f"ALL_cells_{m}_CpG_site_labels.csv")
    stg1_csv = os.path.join(base_dir, f"ALL_cells_{m}_Stage1_site_dataset.csv")
    return long_csv, thr_csv, site_csv, stg1_csv


def compute_cell_stats(vals: np.ndarray, mc_vals: np.ndarray = None) -> dict:
    x = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=float)
    out = {
        "n_obs": int(x.size),
        "hmc_mean": np.nan,
        "hmc_std": np.nan,
        "hmc_median": np.nan,
        "hmc_q10": np.nan,
        "hmc_q25": np.nan,
        "hmc_q50": np.nan,
        "hmc_q75": np.nan,
        "hmc_q85": np.nan,
        "hmc_q90": np.nan,
        "hmc_q95": np.nan,
    }
    if x.size > 0:
        out.update({
            "hmc_mean": float(np.mean(x)),
            "hmc_std": float(np.std(x, ddof=0)),
            "hmc_median": float(np.median(x)),
            "hmc_q10": float(np.quantile(x, 0.10)),
            "hmc_q25": float(np.quantile(x, 0.25)),
            "hmc_q50": float(np.quantile(x, 0.50)),
            "hmc_q75": float(np.quantile(x, 0.75)),
            "hmc_q85": float(np.quantile(x, 0.85)),
            "hmc_q90": float(np.quantile(x, 0.90)),
            "hmc_q95": float(np.quantile(x, 0.95)),
        })

    if mc_vals is not None:
        m = pd.to_numeric(pd.Series(mc_vals), errors="coerce").dropna().to_numpy(dtype=float)
        out["mc_mean"] = float(np.mean(m)) if m.size > 0 else np.nan
        out["mc_std"] = float(np.std(m, ddof=0)) if m.size > 0 else np.nan
        out["mc_q90"] = float(np.quantile(m, 0.90)) if m.size > 0 else np.nan
    else:
        out["mc_mean"] = np.nan
        out["mc_std"] = np.nan
        out["mc_q90"] = np.nan

    return out


def assign_label_cell_by_mode(
    signal: pd.Series,
    thr_gmm: float,
    mode: str = "gmm",
    top_frac: float = 0.15,
    rank_pos_thr: float = 0.85,
    rank_neg_thr: float = 0.50,
    hybrid_rank_thr: float = 0.80,
    hybrid_neg_thr: float = 0.40
) -> pd.DataFrame:
    """
    Return columns:
      - label_cell
      - rank_pct_cell
      - zscore_cell
      - thr_cell_used
      - label_confidence
    """
    s = pd.to_numeric(signal, errors="coerce")
    out = pd.DataFrame(index=s.index)
    out["label_cell"] = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out["rank_pct_cell"] = np.nan
    out["zscore_cell"] = np.nan
    out["thr_cell_used"] = np.nan
    out["label_confidence"] = np.nan

    mask = s.notna()
    if mask.sum() == 0:
        return out

    x = s.loc[mask].astype(float)
    n = len(x)

    rank_desc = x.rank(method="first", ascending=False)
    rank_asc = x.rank(method="first", ascending=True)
    rank_pct_desc = rank_desc / n

    out.loc[mask, "rank_pct_cell"] = (1.0 - rank_pct_desc + 1.0 / n).to_numpy(dtype=float)

    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if sd > 0:
        out.loc[mask, "zscore_cell"] = ((x - mu) / sd).to_numpy(dtype=float)
    else:
        out.loc[mask, "zscore_cell"] = 0.0

    mode = mode.lower().strip()

    if mode == "gmm":
        lab = (x > float(thr_gmm)).astype("int64")
        conf = np.abs(x - float(thr_gmm))
        out.loc[mask, "label_cell"] = pd.Series(lab.values, index=x.index, dtype="Int64")
        out.loc[mask, "thr_cell_used"] = float(thr_gmm)
        out.loc[mask, "label_confidence"] = conf.to_numpy(dtype=float)

    elif mode == "top_frac":
        k_pos = max(1, int(np.ceil(n * float(top_frac))))
        order_desc = x.sort_values(ascending=False, kind="mergesort")
        pos_idx = order_desc.index[:k_pos]

        lab = pd.Series(0, index=x.index, dtype="Int64")
        lab.loc[pos_idx] = 1

        boundary_val = float(order_desc.iloc[k_pos - 1])
        out.loc[mask, "label_cell"] = lab
        out.loc[mask, "thr_cell_used"] = boundary_val

        conf = np.abs(rank_desc - k_pos)
        out.loc[mask, "label_confidence"] = conf.to_numpy(dtype=float)

    elif mode == "rank_hard":
        k_pos = max(1, int(np.ceil(n * (1.0 - float(rank_pos_thr)))))
        k_neg = max(1, int(np.floor(n * float(rank_neg_thr))))

        order_desc = x.sort_values(ascending=False, kind="mergesort")
        order_asc = x.sort_values(ascending=True, kind="mergesort")

        pos_idx = order_desc.index[:k_pos]
        neg_idx = order_asc.index[:k_neg].difference(pos_idx)

        lab = pd.Series(pd.NA, index=x.index, dtype="Int64")
        lab.loc[pos_idx] = 1
        lab.loc[neg_idx] = 0

        out.loc[mask, "label_cell"] = lab
        out.loc[mask, "thr_cell_used"] = np.nan

        conf = np.minimum(np.abs(rank_desc - k_pos), np.abs(rank_asc - k_neg))
        out.loc[mask, "label_confidence"] = conf.to_numpy(dtype=float)

    elif mode == "hybrid_gmm_rank":
        k_pos = max(1, int(np.ceil(n * (1.0 - float(hybrid_rank_thr)))))
        k_neg = max(1, int(np.floor(n * float(hybrid_neg_thr))))

        order_desc = x.sort_values(ascending=False, kind="mergesort")
        order_asc = x.sort_values(ascending=True, kind="mergesort")

        rank_pos_idx = set(order_desc.index[:k_pos])
        rank_neg_idx = set(order_asc.index[:k_neg])

        pos_idx = [idx for idx in x.index if (idx in rank_pos_idx and x.loc[idx] > float(thr_gmm))]
        neg_idx = [idx for idx in x.index if (idx in rank_neg_idx and idx not in pos_idx)]

        lab = pd.Series(pd.NA, index=x.index, dtype="Int64")
        if len(pos_idx) > 0:
            lab.loc[pos_idx] = 1
        if len(neg_idx) > 0:
            lab.loc[neg_idx] = 0

        out.loc[mask, "label_cell"] = lab
        out.loc[mask, "thr_cell_used"] = float(thr_gmm)

        conf = np.minimum(np.abs(rank_desc - k_pos), np.abs(rank_asc - k_neg))
        out.loc[mask, "label_confidence"] = conf.to_numpy(dtype=float)

    else:
        raise ValueError(f"Unsupported --label_cell_mode: {mode}")

    lab_mask = out["label_cell"].notna() & pd.to_numeric(out["label_confidence"], errors="coerce").notna()
    if lab_mask.sum() > 0:
        cc = pd.to_numeric(out.loc[lab_mask, "label_confidence"], errors="coerce")
        lo = cc.min()
        hi = cc.max()
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            out.loc[lab_mask, "label_confidence"] = ((cc - lo) / (hi - lo)).astype(float)
        else:
            out.loc[lab_mask, "label_confidence"] = 1.0

    return out


def summarize_labeling(cell_type: str, part: pd.DataFrame) -> dict:
    y = pd.to_numeric(part["label_cell"], errors="coerce")
    obs = int(y.notna().sum())
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return {
        "cell_type": cell_type,
        "label_obs": obs,
        "label_pos": pos,
        "label_neg": neg,
        "label_pos_rate": float(pos / obs) if obs > 0 else np.nan
    }


def compute_site_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(long_df["label_cell"], errors="coerce").astype(float)
    conf = pd.to_numeric(long_df.get("label_confidence", np.nan), errors="coerce").fillna(0.0).astype(float)

    tmp = pd.DataFrame({
        "chrom": long_df["chrom"].astype(str),
        "pos_based": pd.to_numeric(long_df["pos_based"], errors="coerce"),
        "y": y,
        "conf": conf,
    })

    is_pos = tmp["y"].eq(1).fillna(False).to_numpy(dtype=bool)
    is_neg = tmp["y"].eq(0).fillna(False).to_numpy(dtype=bool)
    is_obs = tmp["y"].notna().to_numpy(dtype=bool)
    conf_np = tmp["conf"].to_numpy(dtype=float)

    tmp["pos_conf"] = np.where(is_pos, conf_np, 0.0)
    tmp["neg_conf"] = np.where(is_neg, conf_np, 0.0)
    tmp["obs_conf"] = np.where(is_obs, conf_np, 0.0)

    st = (
        tmp.groupby(["chrom", "pos_based"], as_index=False)
        .agg(
            pos_count=("y", lambda s: int(np.nansum(s.to_numpy(dtype=float) == 1))),
            obs_count=("y", lambda s: int(np.sum(~np.isnan(s.to_numpy(dtype=float))))),
            pos_conf_sum=("pos_conf", "sum"),
            neg_conf_sum=("neg_conf", "sum"),
            obs_conf_sum=("obs_conf", "sum"),
        )
    )
    st["pos_ratio"] = st["pos_count"] / st["obs_count"].replace(0, np.nan)
    st["pos_conf_ratio"] = st["pos_conf_sum"] / st["obs_conf_sum"].replace(0, np.nan)
    st["site_soft_score"] = (
        0.5 * st["pos_ratio"].fillna(0.0) + 0.5 * st["pos_conf_ratio"].fillna(0.0)
    ).astype(float)
    return st


def assign_label_site(site_df: pd.DataFrame,
                      mode: str,
                      ratio_r: float,
                      n_pos: int,
                      min_obs: int,
                      neg_mode: str,
                      conf_ratio_thr: float = 0.55,
                      pos_conf_min: float = 1.50) -> pd.DataFrame:
    df = site_df.copy()
    df["label_site"] = pd.NA

    ok = df["obs_count"] >= int(min_obs)

    mode = mode.lower().strip()
    if mode == "ratio":
        pos_mask = ok & (df["pos_ratio"] >= float(ratio_r))
    elif mode == "npos":
        pos_mask = ok & (df["pos_count"] >= int(n_pos))
    elif mode == "npos_weighted":
        pos_conf_ratio = pd.to_numeric(df.get("pos_conf_ratio", 0.0), errors="coerce").fillna(0.0)
        pos_conf_sum = pd.to_numeric(df.get("pos_conf_sum", 0.0), errors="coerce").fillna(0.0)
        pos_mask = (
            ok
            & (df["pos_count"] >= int(n_pos))
            & (pos_conf_ratio >= float(conf_ratio_thr))
            & (pos_conf_sum >= float(pos_conf_min))
        )
    else:
        raise ValueError("--site_label_mode must be ratio, npos, or npos_weighted")

    neg_mode = neg_mode.lower().strip()
    if neg_mode == "strict":
        neg_mask = ok & (df["pos_count"] == 0)
    elif neg_mode == "loose":
        neg_mask = (df["obs_count"] > 0) & (df["pos_count"] == 0)
    else:
        raise ValueError("--neg_mode must be strict or loose")

    df["site_role"] = "uncertain"
    df.loc[pos_mask, "label_site"] = 1
    df.loc[neg_mask, "label_site"] = 0
    df.loc[pos_mask, "site_role"] = "pos"
    df.loc[neg_mask, "site_role"] = "neg"
    return df


def build_stage1_dataset(site_labeled: pd.DataFrame,
                         neg_pos_ratio: float,
                         seed: int) -> pd.DataFrame:
    pos_df = site_labeled[site_labeled["site_role"] == "pos"].copy()
    neg_df = site_labeled[site_labeled["site_role"] == "neg"].copy()

    n_pos = len(pos_df)
    if n_pos == 0:
        return site_labeled.head(0).copy()

    target_neg = int(np.ceil(float(neg_pos_ratio) * n_pos))
    if len(neg_df) > target_neg:
        neg_df = neg_df.sample(n=target_neg, random_state=int(seed))

    out = pd.concat([pos_df, neg_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)
    return out

def print_figure2_stats_only(df_wide, long_df, site_df, cell_types, dataset_name="Dataset"):
    """
    只打印 Figure 2 需要的统计结果，不保存文件。

    输入对象来自 P1 当前流程：
      df_wide: P1 读取的宽表，也就是 df
      long_df: P1 构建出的 site × cell long table
      site_df: assign_label_site() 后、删除 uncertain 前的 site-level table
      cell_types: 当前数据集的细胞群列表
    """

    print("\n" + "=" * 100)
    print(f"FIGURE 2 SOURCE STATISTICS ONLY | {dataset_name}")
    print("=" * 100)

    # ============================================================
    # Figure 2A: Data sources and observed CpG sites
    # ============================================================
    print("\n[Figure 2A] Observed CpG sites per cell group")
    print("-" * 100)

    rows_a = []
    total_sites = len(df_wide)

    for cell in cell_types:
        hmc_col = f"5hmc_signal_{cell}"
        mc_col = f"5mc_signal_{cell}"

        if hmc_col not in df_wide.columns:
            print(f"WARNING: missing column {hmc_col}")
            continue

        hmc = pd.to_numeric(df_wide[hmc_col], errors="coerce")
        mc = pd.to_numeric(df_wide[mc_col], errors="coerce") if mc_col in df_wide.columns else pd.Series(dtype=float)

        obs_hmc = int(hmc.notna().sum())
        obs_mc = int(mc.notna().sum()) if len(mc) else np.nan
        hmc_nonzero = int((hmc > 0).sum())

        rows_a.append({
            "dataset": dataset_name,
            "cell_type": cell,
            "candidate_union_sites": total_sites,
            "observed_hmc_sites": obs_hmc,
            "observed_hmc_million": obs_hmc / 1e6,
            "observed_hmc_rate": obs_hmc / total_sites if total_sites > 0 else np.nan,
            "hmc_nonzero_sites": hmc_nonzero,
            "hmc_nonzero_rate_in_observed": hmc_nonzero / obs_hmc if obs_hmc > 0 else np.nan,
            "observed_mc_sites": obs_mc
        })

    fig2a = pd.DataFrame(rows_a)
    print(fig2a.to_string(index=False))

    print(
        "\nNOTE: Samples(n) 不能从当前 P0/P1 的四列表输入直接统计。"
        "当前输入只包含 chrom / pos_c_1based / 5hmc_signal / 5mc_signal。"
    )

    # ============================================================
    # Figure 2B: label-state proportions
    # ============================================================
    print("\n[Figure 2B] Label-state proportions")
    print("-" * 100)

    site_keep_cols = [
        "chrom", "pos_based", "site_role", "label_site",
        "pos_count", "obs_count", "pos_ratio",
        "pos_conf_sum", "neg_conf_sum", "obs_conf_sum",
        "pos_conf_ratio", "site_soft_score"
    ]
    site_keep_cols = [c for c in site_keep_cols if c in site_df.columns]

    x = long_df.merge(
        site_df[site_keep_cols],
        on=["chrom", "pos_based"],
        how="left"
    ).copy()

    y = pd.to_numeric(x["label_cell"], errors="coerce")

    # Figure 2B 中的三类：
    # positive: retained site 且 cell-level label = 1
    # negative: retained site 且 cell-level label = 0
    # missing_filtered: uncertain site 或 cell-level label 缺失
    retained = x["site_role"].isin(["pos", "neg"])

    x["fig2_state"] = "missing_filtered"
    x.loc[retained & y.eq(1), "fig2_state"] = "positive"
    x.loc[retained & y.eq(0), "fig2_state"] = "negative"

    fig2b = (
        x.groupby(["cell_type", "fig2_state"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["positive", "negative", "missing_filtered"]:
        if col not in fig2b.columns:
            fig2b[col] = 0

    fig2b["total"] = fig2b["positive"] + fig2b["negative"] + fig2b["missing_filtered"]

    fig2b["positive_prop"] = fig2b["positive"] / fig2b["total"].replace(0, np.nan)
    fig2b["negative_prop"] = fig2b["negative"] / fig2b["total"].replace(0, np.nan)
    fig2b["missing_filtered_prop"] = fig2b["missing_filtered"] / fig2b["total"].replace(0, np.nan)

    fig2b = fig2b[
        [
            "cell_type",
            "positive", "negative", "missing_filtered", "total",
            "positive_prop", "negative_prop", "missing_filtered_prop"
        ]
    ]

    print(fig2b.to_string(index=False))

    print("\n[Figure 2B] Overall state proportions")
    overall = x["fig2_state"].value_counts().rename_axis("state").reset_index(name="count")
    overall["prop"] = overall["count"] / overall["count"].sum()
    print(overall.to_string(index=False))

    # ============================================================
    # Figure 2C: site-level label construction statistics
    # ============================================================
    print("\n[Figure 2C] Site-level label construction statistics")
    print("-" * 100)

    print("\nSite role counts:")
    print(site_df["site_role"].value_counts(dropna=False).to_string())

    print("\nSite role proportions:")
    print((site_df["site_role"].value_counts(dropna=False) / len(site_df)).to_string())

    print("\nobs_count distribution:")
    print(site_df["obs_count"].value_counts().sort_index().to_string())

    print("\npos_count distribution:")
    print(site_df["pos_count"].value_counts().sort_index().to_string())

    print("\nsite-level summary:")
    summary_rows = []
    for role in ["pos", "neg", "uncertain"]:
        sub = site_df[site_df["site_role"] == role]
        summary_rows.append({
            "site_role": role,
            "n_sites": len(sub),
            "prop_sites": len(sub) / len(site_df) if len(site_df) else np.nan,
            "mean_obs_count": sub["obs_count"].mean() if len(sub) else np.nan,
            "median_obs_count": sub["obs_count"].median() if len(sub) else np.nan,
            "mean_pos_count": sub["pos_count"].mean() if len(sub) else np.nan,
            "median_pos_count": sub["pos_count"].median() if len(sub) else np.nan,
            "mean_pos_ratio": sub["pos_ratio"].mean() if len(sub) else np.nan,
            "median_pos_ratio": sub["pos_ratio"].median() if len(sub) else np.nan,
        })
    fig2c_summary = pd.DataFrame(summary_rows)
    print(fig2c_summary.to_string(index=False))

    # ============================================================
    # Figure 2C: real example sites for schematic drawing
    # ============================================================
    print("\n[Figure 2C] Example real sites for schematic")
    print("-" * 100)

    example_parts = []

    pos_ex = (
        site_df[site_df["site_role"] == "pos"]
        .sort_values(["pos_count", "pos_ratio", "obs_count"], ascending=[False, False, False])
        .head(3)
        .copy()
    )
    pos_ex["example_type"] = "recurrent_positive"

    neg_ex = (
        site_df[site_df["site_role"] == "neg"]
        .sort_values(["obs_count", "pos_count"], ascending=[False, True])
        .head(3)
        .copy()
    )
    neg_ex["example_type"] = "mostly_negative"

    filt_ex = (
        site_df[site_df["site_role"] == "uncertain"]
        .sort_values(["obs_count", "pos_count"], ascending=[True, True])
        .head(3)
        .copy()
    )
    filt_ex["example_type"] = "filtered_uncertain"

    example_parts = [pos_ex, neg_ex, filt_ex]
    examples = pd.concat(example_parts, ignore_index=True)

    if len(examples) == 0:
        print("No example sites found.")
    else:
        ex_sites = examples[["chrom", "pos_based", "example_type", "site_role", "pos_count", "obs_count", "pos_ratio"]].copy()

        ex_long = x.merge(
            ex_sites[["chrom", "pos_based", "example_type"]],
            on=["chrom", "pos_based"],
            how="inner"
        ).copy()

        label_num = pd.to_numeric(ex_long["label_cell"], errors="coerce")
        ex_long["label_symbol"] = "."
        ex_long.loc[label_num.eq(1), "label_symbol"] = "+"
        ex_long.loc[label_num.eq(0), "label_symbol"] = "-"

        label_matrix = (
            ex_long.pivot_table(
                index=["example_type", "chrom", "pos_based"],
                columns="cell_type",
                values="label_symbol",
                aggfunc="first"
            )
            .reset_index()
        )

        # 按 cell_types 固定列顺序
        ordered_cols = ["example_type", "chrom", "pos_based"] + [c for c in cell_types if c in label_matrix.columns]
        label_matrix = label_matrix[ordered_cols]

        print("\nExample site label matrix:")
        print(label_matrix.to_string(index=False))

        print("\nExample site statistics:")
        print(ex_sites.to_string(index=False))

    print("\n" + "=" * 100)
    print("END OF FIGURE 2 STATISTICS")
    print("=" * 100 + "\n")
def main():
    ap = argparse.ArgumentParser(description="P1: wide->long + label_cell + label_site (+ Stage1 sampling)")
    ap.add_argument("--base_dir", default=r"data/data_out/Brain_CPG_ALL_UNION")
    ap.add_argument("--input_mode", choices=["union", "intersection"], default="union")
    ap.add_argument("--input_csv", default="", help="optional manual input csv path")

    ap.add_argument("--p0_high_conf_k", type=int, default=None,
                    help="if set, use ALL_cells_<mode>_CpG_4cols_high_conf_hmc{k}.csv as input")

    ap.add_argument("--min_gmm_samples", type=int, default=500)
    ap.add_argument("--fallback_thr", type=float, default=0.1)

    ap.add_argument("--label_cell_mode",
                    choices=["gmm", "top_frac", "rank_hard", "hybrid_gmm_rank"],
                    default="gmm",
                    help="how to define label_cell for each cell")
    ap.add_argument("--cell_pos_frac", type=float, default=0.15)
    ap.add_argument("--rank_pos_thr", type=float, default=0.85)
    ap.add_argument("--rank_neg_thr", type=float, default=0.50)
    ap.add_argument("--hybrid_rank_thr", type=float, default=0.80)
    ap.add_argument("--hybrid_neg_thr", type=float, default=0.40)

    ap.add_argument("--site_label_mode", choices=["ratio", "npos", "npos_weighted"], default="ratio")
    ap.add_argument("--site_ratio_r", type=float, default=0.30)
    ap.add_argument("--site_n_pos", type=int, default=3)
    ap.add_argument("--site_conf_ratio_thr", type=float, default=0.55)
    ap.add_argument("--site_pos_conf_min", type=float, default=1.50)
    ap.add_argument("--min_obs_cells", type=int, default=5)
    ap.add_argument("--neg_mode", choices=["strict", "loose"], default="strict")

    ap.add_argument("--stage1_neg_pos_ratio", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--scan_grid", action="store_true",
                    help="print grid scan over (min_obs=m) and (npos=n) to help pick stable settings")
    ap.add_argument("--dataset_name", default="Dataset")

    args = ap.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)

    if args.input_csv.strip():
        in_path = args.input_csv.strip()
        mode = "custom"
    else:
        in_path = build_input_path(args.base_dir, args.input_mode, args.p0_high_conf_k)
        mode = args.input_mode

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")

    long_csv, thr_csv, site_csv, stg1_csv = default_outputs(args.base_dir, args.input_mode)

    print("=" * 100)
    print("🧩 P1: wide -> long + labels")
    print("=" * 100)
    print(f"Input : {in_path}")
    print(f"Mode  : {mode}")
    print(f"GMM   : min_samples={args.min_gmm_samples} fallback_thr={args.fallback_thr}")
    print(f"label_cell: mode={args.label_cell_mode} "
          f"top_frac={args.cell_pos_frac} "
          f"rank_pos={args.rank_pos_thr} rank_neg={args.rank_neg_thr} "
          f"hybrid_rank={args.hybrid_rank_thr} hybrid_neg={args.hybrid_neg_thr}")
    print(f"label_site: mode={args.site_label_mode} r={args.site_ratio_r} "
          f"n_pos={args.site_n_pos} conf_ratio={args.site_conf_ratio_thr} pos_conf_min={args.site_pos_conf_min} "
          f"min_obs={args.min_obs_cells} neg_mode={args.neg_mode}")
    print(f"Stage1 sampling: neg:pos={args.stage1_neg_pos_ratio}:1")
    print("Outputs:")
    print(f"  - long : {long_csv}")
    print(f"  - thr  : {thr_csv}")
    print(f"  - site : {site_csv}")
    print(f"  - stg1 : {stg1_csv}")

    df = pd.read_csv(in_path, low_memory=False)
    print(f"\n✅ Wide table loaded: {df.shape}")

    for c in ["chrom", "pos_based"]:
        if c not in df.columns:
            raise ValueError(f"Missing required col: {c}")

    hmc_cols = [c for c in df.columns if c.startswith("5hmc_signal_")]
    mc_cols = [c for c in df.columns if c.startswith("5mc_signal_")]

    if not hmc_cols:
        raise ValueError("No 5hmc_signal_* columns found.")
    if not mc_cols:
        print("⚠️ No 5mc_signal_* columns found. Will fill 5mc_signal=NaN in long table.")

    cell_types = [c.replace("5hmc_signal_", "") for c in hmc_cols]
    print(f"✅ Detected cell_types={len(cell_types)}: {cell_types}")

    thresholds = {}
    cell_stats_rows = []

    for cell in cell_types:
        vals = pd.to_numeric(df[f"5hmc_signal_{cell}"], errors="coerce").to_numpy()
        mc_col = f"5mc_signal_{cell}"
        mc_vals = pd.to_numeric(df[mc_col], errors="coerce").to_numpy() if mc_col in df.columns else None

        thr = float(fit_gmm_threshold(vals, min_samples=args.min_gmm_samples, fallback=args.fallback_thr))
        thresholds[cell] = thr

        st = compute_cell_stats(vals, mc_vals=mc_vals)
        st["cell_type"] = cell
        st["thr_cell_gmm"] = thr
        cell_stats_rows.append(st)

    cell_stats_df = pd.DataFrame(cell_stats_rows)
    cell_stats_df = cell_stats_df[[
        "cell_type", "thr_cell_gmm",
        "n_obs", "hmc_mean", "hmc_std", "hmc_median",
        "hmc_q10", "hmc_q25", "hmc_q50", "hmc_q75", "hmc_q85", "hmc_q90", "hmc_q95",
        "mc_mean", "mc_std", "mc_q90"
    ]]
    cell_stats_df.to_csv(thr_csv, index=False)
    print(f"✅ Saved thresholds/cell stats: {thr_csv}")
    print(cell_stats_df.head().to_string(index=False))

    base = df[["chrom", "pos_based"]].copy()
    base["chrom"] = base["chrom"].astype(str)
    base["pos_based"] = pd.to_numeric(base["pos_based"], errors="coerce")

    long_parts = []
    for cell in cell_types:
        part = base.copy()
        part["cell_type"] = cell
        part["5hmc_signal"] = pd.to_numeric(df[f"5hmc_signal_{cell}"], errors="coerce")

        mc_col = f"5mc_signal_{cell}"
        part["5mc_signal"] = pd.to_numeric(df[mc_col], errors="coerce") if mc_col in df.columns else np.nan

        thr = thresholds[cell]
        part["thr_cell_gmm"] = float(thr)

        lab_df = assign_label_cell_by_mode(
            signal=part["5hmc_signal"],
            thr_gmm=thr,
            mode=args.label_cell_mode,
            top_frac=args.cell_pos_frac,
            rank_pos_thr=args.rank_pos_thr,
            rank_neg_thr=args.rank_neg_thr,
            hybrid_rank_thr=args.hybrid_rank_thr,
            hybrid_neg_thr=args.hybrid_neg_thr
        )

        part["label_cell"] = lab_df["label_cell"]
        part["rank_pct_cell"] = lab_df["rank_pct_cell"]
        part["zscore_cell"] = lab_df["zscore_cell"]
        part["thr_cell_used"] = lab_df["thr_cell_used"]
        part["label_confidence"] = lab_df["label_confidence"]
        part["label_cell_mode"] = args.label_cell_mode

        st_row = cell_stats_df[cell_stats_df["cell_type"] == cell].iloc[0].to_dict()
        for k, v in st_row.items():
            if k != "cell_type":
                part[k] = v

        long_parts.append(part)

    long_df = pd.concat(long_parts, ignore_index=True)

    # 全量 long_df 先做一次 site stats
    site_df = compute_site_stats(long_df)

    print("\n==================== obs_count 分布（位点覆盖度） ====================")
    obs_vc = site_df["obs_count"].value_counts().sort_index()
    print(obs_vc.to_string())
    print("obs_count 平均值:", float(site_df["obs_count"].mean()))
    print("obs_count 中位数:", float(site_df["obs_count"].median()))

    if args.scan_grid:
        print("\n==================== 网格扫描 (min_obs=m, n_pos=n) ====================")
        ms = [3, 4, 5, 6, 7]
        ns = [1, 2, 3, 4]
        for m_test in ms:
            for n_test in ns:
                ok = site_df["obs_count"] >= m_test
                pos = int((ok & (site_df["pos_count"] >= n_test)).sum())
                neg_strict = int((ok & (site_df["pos_count"] == 0)).sum())
                unc = len(site_df) - pos - neg_strict
                print(f"m={m_test} n={n_test} | pos={pos:>8,} neg={neg_strict:>8,} uncertain={unc:>8,}")
        print("======================================================================\n")

    site_df = assign_label_site(
        site_df,
        mode=args.site_label_mode,
        ratio_r=args.site_ratio_r,
        n_pos=args.site_n_pos,
        min_obs=args.min_obs_cells,
        neg_mode=args.neg_mode,
        conf_ratio_thr=args.site_conf_ratio_thr,
        pos_conf_min=args.site_pos_conf_min
    )

    print("\n==================== 原始 site_role 统计 ====================")
    print(site_df["site_role"].value_counts().to_dict())

    print_figure2_stats_only(
        df_wide=df,
        long_df=long_df,
        site_df=site_df,
        cell_types=cell_types,
        dataset_name=args.dataset_name if hasattr(args, "dataset_name") else "Dataset"
    )

    # ===== 关键修改：直接删除 uncertain =====
    keep_sites = (
        site_df.loc[site_df["site_role"].isin(["pos", "neg"]), ["chrom", "pos_based"]]
        .drop_duplicates()
        .copy()
    )

    long_df = long_df.merge(keep_sites, on=["chrom", "pos_based"], how="inner").copy()
    site_df = site_df.merge(keep_sites, on=["chrom", "pos_based"], how="inner").copy()

    # 过滤后重新统计每个 cell 的 labeling summary
    label_sum_rows = []
    for cell in cell_types:
        sub = long_df[long_df["cell_type"] == cell].copy()
        label_sum_rows.append(summarize_labeling(cell, sub))
    label_sum_df = pd.DataFrame(label_sum_rows)

    cell_stats_df = cell_stats_df.merge(label_sum_df, on="cell_type", how="left")
    cell_stats_df.to_csv(thr_csv, index=False)

    long_df.to_csv(long_csv, index=False, na_rep="NaN")
    print(f"\n✅ Saved FILTERED long table: {long_csv}")
    print(f"   rows={len(long_df):,} unique_sites={long_df[['chrom','pos_based']].drop_duplicates().shape[0]:,}")
    print(f"   label_cell non-NA rate={float(long_df['label_cell'].notna().mean()):.2%}")

    print("\n==================== label_cell summary by cell (filtered sites) ====================")
    print(label_sum_df.to_string(index=False))

    site_df.to_csv(site_csv, index=False, na_rep="NaN")
    print(f"\n✅ Saved FILTERED site labels: {site_csv}")
    print("   filtered site_role counts:", site_df["site_role"].value_counts().to_dict())

    stg1_df = build_stage1_dataset(site_df, neg_pos_ratio=args.stage1_neg_pos_ratio, seed=args.seed)
    if len(stg1_df) == 0:
        print("⚠️ Stage1 dataset empty (no positives or too strict).")
    else:
        keep_cols = [c for c in [
            "chrom", "pos_based", "label_site", "site_role",
            "pos_count", "obs_count", "pos_ratio",
            "pos_conf_sum", "neg_conf_sum", "obs_conf_sum",
            "pos_conf_ratio", "site_soft_score"
        ] if c in stg1_df.columns]
        stg1_out = stg1_df[keep_cols].copy()
        stg1_out.to_csv(stg1_csv, index=False, na_rep="NaN")
        n_pos = int((stg1_out["site_role"] == "pos").sum())
        n_neg = int((stg1_out["site_role"] == "neg").sum())
        print(f"✅ Saved Stage1 site dataset: {stg1_csv}")
        print(f"   pos={n_pos:,} neg={n_neg:,} (neg:pos={n_neg/max(1,n_pos):.2f}:1)")

    print("\n🎉 P1 finished.")


if __name__ == "__main__":
    main()