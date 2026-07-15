#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P4_build_stage2_dataset.py
Stage4b: Build Stage-2 (cell) dataset (site×cell) by merging:
  - P2b long_with_gene (site×cell keys + label_cell)
  - P2a static seq features (site-level)
  - P2c TF motif features (gene-level)
  - P3a dynamic site mC features (site×cell or site-level)
  - P3b promoter mC (gene×cell)
  - relative/coupling/domain features for better LOCTO

Modified version:
  - keep label_confidence as training-only sample_weight (not as feature)
  - add a few safe interaction features for Stage2
  - preserve original leakage cleanup for 5hmc-derived row-level columns

Output:
  - model_input/.../stage2_cell_static_dynamic_{mode}.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

CELL_TYPES = ["OPC", "ODC1", "ODC2", "ODC3", "MGC", "INH", "ENDO", "ASC1", "ASC2", "EXC1", "EXC2"]
# CELL_TYPES = ["2i", "serum"]
# CELL_TYPES = ["B", "T_reg", "T_naive", "NK", "Monocytes"]
MC_COLS = [f"mC_{c}" for c in CELL_TYPES]


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _ensure_cols(df: pd.DataFrame, cols, name="df"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def norm_gid(g: str) -> str:
    g = str(g).strip()
    if g in {"", "nan", "None"}:
        return ""
    return g.split(".")[0]


def _pick_this_cell_value(df: pd.DataFrame, cell_col: str, value_cols: list, out_name: str) -> pd.Series:
    col_map = {ct: f"mC_{ct}" for ct in CELL_TYPES}
    for c in col_map.values():
        if c not in df.columns:
            raise ValueError(f"Cannot build {out_name}: missing column {c}")

    vals = df[[col_map[ct] for ct in CELL_TYPES]].to_numpy(dtype=np.float32, copy=False)
    idx = df[cell_col].map({ct: i for i, ct in enumerate(CELL_TYPES)}).fillna(-1).astype(int).to_numpy()
    out = np.full(len(df), np.nan, dtype=np.float32)
    ok = idx >= 0
    out[ok] = vals[np.arange(len(df))[ok], idx[ok]]
    return pd.Series(out, name=out_name)


def _rename_cell_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only safe cell-level domain features from P1 long table.
    These are cell-level summaries, not per-row label surrogates.
    """
    rename_map = {
        "thr_cell_gmm": "cell_thr_gmm",
        "n_obs": "cell_hmc_n_obs",
        "hmc_mean": "cell_hmc_mean",
        "hmc_std": "cell_hmc_std",
        "hmc_median": "cell_hmc_median",
        "hmc_q10": "cell_hmc_q10",
        "hmc_q25": "cell_hmc_q25",
        "hmc_q50": "cell_hmc_q50",
        "hmc_q75": "cell_hmc_q75",
        "hmc_q85": "cell_hmc_q85",
        "hmc_q90": "cell_hmc_q90",
        "hmc_q95": "cell_hmc_q95",
        "mc_mean": "cell_mc_mean",
        "mc_std": "cell_mc_std",
        "mc_q90": "cell_mc_q90",
    }
    keep = [c for c in rename_map if c in df.columns]
    if keep:
        df = df.rename(columns={c: rename_map[c] for c in keep})
    return df


def _build_sample_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve label confidence for training-time weighting only.
    Do NOT keep label_confidence as a normal feature.
    """
    out = df.copy()
    if "label_confidence" not in out.columns:
        out["sample_weight"] = np.float32(1.0)
        return out

    conf = pd.to_numeric(out["label_confidence"], errors="coerce")
    valid = conf.notna()
    if valid.sum() == 0:
        out["sample_weight"] = np.float32(1.0)
        return out

    lo = conf[valid].min()
    hi = conf[valid].max()
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        # map to [0.5, 1.5] to avoid zero/near-zero weights
        scaled = 0.5 + (conf - lo) / (hi - lo)
    else:
        scaled = pd.Series(1.0, index=out.index)

    out["sample_weight"] = scaled.fillna(1.0).astype("float32")
    return out


def _add_promoter_relative_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    long_df must already contain promoter mC columns mC_OPC..mC_EXC2
    """
    df = long_df.copy()

    m = df[MC_COLS].mean(axis=1, skipna=True).astype("float32")
    s = df[MC_COLS].std(axis=1, skipna=True).fillna(0.0).astype("float32")

    df["mC_promoter_mean_cells"] = m
    df["mC_promoter_std_cells"] = s
    df["mC_promoter_this_cell"] = _pick_this_cell_value(df, "cell_type", MC_COLS, "mC_promoter_this_cell").astype("float32")

    df["mC_promoter_delta"] = (df["mC_promoter_this_cell"] - df["mC_promoter_mean_cells"]).astype("float32")
    df["mC_promoter_z"] = np.where(
        df["mC_promoter_std_cells"] > 0,
        df["mC_promoter_delta"] / df["mC_promoter_std_cells"],
        0.0
    ).astype("float32")

    rank_pct = df[MC_COLS].rank(axis=1, method="average", pct=True)
    rank_input = pd.concat([df[["cell_type"]], rank_pct], axis=1)
    df["mC_promoter_rank_pct"] = _pick_this_cell_value(rank_input, "cell_type", MC_COLS, "mC_promoter_rank_pct").astype("float32")

    return df


def _add_site_relative_features_from_p3a(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix bug: P3a provides mC_site_this_cell (not mC_this_cell).
    Add richer site-relative features.
    """
    df = long_df.copy()

    needed = ["mC_site_this_cell", "mC_site_mean", "mC_site_std"]
    if all(c in df.columns for c in needed):
        x = pd.to_numeric(df["mC_site_this_cell"], errors="coerce")
        m = pd.to_numeric(df["mC_site_mean"], errors="coerce")
        s = pd.to_numeric(df["mC_site_std"], errors="coerce").fillna(0.0)

        df["mC_site_delta"] = (x - m).astype("float32")
        df["mC_site_z"] = np.where(s > 0, df["mC_site_delta"] / s, 0.0).astype("float32")

        if "mC_site_range" in df.columns:
            rng = pd.to_numeric(df["mC_site_range"], errors="coerce")
            df["mC_site_range_over_mean"] = np.where(
                np.abs(m) > 1e-6, rng / (np.abs(m) + 1e-6), 0.0
            ).astype("float32")

        if "mC_site_top2gap" in df.columns:
            gap = pd.to_numeric(df["mC_site_top2gap"], errors="coerce")
            df["mC_site_top2gap_over_std"] = np.where(
                s > 0, gap / (s + 1e-6), 0.0
            ).astype("float32")

    return df


def _add_site_promoter_coupling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Couple local site mC and promoter mC.
    """
    out = df.copy()

    need1 = ["mC_site_this_cell", "mC_promoter_this_cell"]
    if all(c in out.columns for c in need1):
        site = pd.to_numeric(out["mC_site_this_cell"], errors="coerce")
        prom = pd.to_numeric(out["mC_promoter_this_cell"], errors="coerce")
        out["mC_site_minus_promoter"] = (site - prom).astype("float32")
        out["mC_site_over_promoter"] = np.where(
            np.abs(prom) > 1e-6, site / (prom + 1e-6), 0.0
        ).astype("float32")

    need2 = ["mC_site_z", "mC_promoter_z"]
    if all(c in out.columns for c in need2):
        out["mC_site_z_minus_promoter_z"] = (
            pd.to_numeric(out["mC_site_z"], errors="coerce") -
            pd.to_numeric(out["mC_promoter_z"], errors="coerce")
        ).astype("float32")

    if "mC_promoter_rank_pct" in out.columns and "mC_site_delta" in out.columns:
        out["mC_site_delta_x_promoter_rank"] = (
            pd.to_numeric(out["mC_site_delta"], errors="coerce") *
            pd.to_numeric(out["mC_promoter_rank_pct"], errors="coerce")
        ).astype("float32")

    return out


def _add_stage2_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few safe interaction features.
    Keep them generic; cell_type one-hot will be done at training stage.
    """
    out = df.copy()

    pairs = [
        ("mC_site_delta", "mC_promoter_z", "mC_site_delta_x_promoter_z"),
        ("mC_site_z", "mC_promoter_rank_pct", "mC_site_z_x_promoter_rank"),
        ("mC_site_this_cell", "mC_promoter_this_cell", "mC_site_x_promoter_this_cell"),
        ("cell_hmc_mean", "mC_site_this_cell", "cell_hmc_mean_x_site_mC"),
        ("cell_mc_mean", "mC_site_this_cell", "cell_mc_mean_x_site_mC"),
    ]

    for c1, c2, out_name in pairs:
        if c1 in out.columns and c2 in out.columns:
            x1 = pd.to_numeric(out[c1], errors="coerce")
            x2 = pd.to_numeric(out[c2], errors="coerce")
            out[out_name] = (x1 * x2).astype("float32")

    if "mC_site_this_cell" in out.columns and "mC_site_mean" in out.columns:
        x = pd.to_numeric(out["mC_site_this_cell"], errors="coerce")
        m = pd.to_numeric(out["mC_site_mean"], errors="coerce")
        out["mC_site_this_over_mean"] = np.where(np.abs(m) > 1e-6, x / (m + 1e-6), 0.0).astype("float32")

    if "mC_promoter_this_cell" in out.columns and "mC_promoter_mean_cells" in out.columns:
        x = pd.to_numeric(out["mC_promoter_this_cell"], errors="coerce")
        m = pd.to_numeric(out["mC_promoter_mean_cells"], errors="coerce")
        out["mC_promoter_this_over_mean"] = np.where(np.abs(m) > 1e-6, x / (m + 1e-6), 0.0).astype("float32")

    return out


def main():
    ap = argparse.ArgumentParser(description="P4 Stage-2 dataset builder (STATIC + DYNAMIC, site×cell)")
    ap.add_argument("--base_dir", default=r"data/data_out/Brain_CPG_ALL_UNION")
    ap.add_argument("--feat_dir", default=r"feature_output/Brain_CPG_ALL_UNION")
    ap.add_argument("--model_in_dir", default=r"model_input/Brain_CPG_ALL_UNION")
    ap.add_argument("--input_mode", choices=["union", "intersection"], default="union")

    ap.add_argument("--use_tf", action="store_true")
    ap.add_argument("--tf_topk", type=int, default=100)

    ap.add_argument("--use_p3a_site_mC", action="store_true")
    ap.add_argument("--use_p3b_promoter_mC", action="store_true")
    ap.add_argument("--p3b_promoter_mc_csv", default=r"data\data_out\Brain_CPG_ALL_UNION\Brain_gene_promoter_5mC_11celltypes.csv")

    ap.add_argument("--add_relative_mC_features", action="store_true",
                    help="add delta/z/rank/coupling features for better LOCTO")
    ap.add_argument("--keep_sample_weight", action="store_true",
                    help="convert label_confidence to sample_weight for training-time use")
    ap.add_argument("--add_stage2_interactions", action="store_true",
                    help="add safe interaction features for Stage2")

    args = ap.parse_args()
    mode = args.input_mode

    long_with_gene = os.path.join(args.base_dir, f"ALL_cells_{mode}_CpG_long_table_with_gene.csv")
    seq_feat_csv   = os.path.join(args.feat_dir, f"seq_features_from_{mode}.csv")
    tf_csv         = os.path.join(args.feat_dir, f"gene_TFlist_maxscore_topK{args.tf_topk}.csv")
    p3a_csv        = os.path.join(args.feat_dir, f"dynamic_site_mC_from_{mode}.csv")

    out_csv = os.path.join(args.model_in_dir, f"stage2_cell_static_dynamic_{mode}.csv")
    os.makedirs(args.model_in_dir, exist_ok=True)

    print("=" * 100)
    print("📦 P4 Stage-2 dataset builder (STATIC + DYNAMIC, site×cell)")
    print("=" * 100)
    print(f"mode: {mode}")
    print(f"P2b long_with_gene: {long_with_gene}")
    print(f"P2a seq features : {seq_feat_csv}")
    print(f"use_tf: {args.use_tf} | use_p3a_site_mC: {args.use_p3a_site_mC} | use_p3b_promoter_mC: {args.use_p3b_promoter_mC}")
    print(f"keep_sample_weight: {args.keep_sample_weight} | add_stage2_interactions: {args.add_stage2_interactions}")
    print(f"Output: {out_csv}")

    # ---------- load long ----------
    df = _read_csv(long_with_gene)
    _ensure_cols(df, ["chrom", "pos_based", "cell_type", "label_cell"], "long_with_gene")

    # keep only labeled rows for stage2
    df["label_cell"] = pd.to_numeric(df["label_cell"], errors="coerce")
    df = df[df["label_cell"].notna()].copy()
    df["label_cell"] = df["label_cell"].astype(int)

    # rename safe cell-level covariates
    df = _rename_cell_covariates(df)

    # preserve training-only weight before leakage cleanup
    if args.keep_sample_weight:
        df = _build_sample_weight(df)
        print("Added training-only sample_weight from label_confidence")

    # ---------- merge seq ----------
    seq = _read_csv(seq_feat_csv)
    _ensure_cols(seq, ["chrom", "pos_based"], "seq_features")
    df = df.merge(seq, on=["chrom", "pos_based"], how="left")
    print(f"After merge seq: {df.shape}")

    # ---------- merge TF ----------
    if args.use_tf:
        tf = _read_csv(tf_csv)
        _ensure_cols(tf, ["gene_id"], "tf_features")

        df["gene_id_base"] = df["gene_id"].astype(str).map(norm_gid)
        tf["gene_id_base"] = tf["gene_id"].astype(str).map(norm_gid)

        drop_cols = [c for c in ["gene_id", "gene_name"] if c in tf.columns]
        tf = tf.drop(columns=drop_cols, errors="ignore").drop_duplicates(subset=["gene_id_base"])

        df = df.merge(tf, on="gene_id_base", how="left")
        print(f"After merge TF: {df.shape}")

    # ---------- merge P3a ----------
    if args.use_p3a_site_mC:
        p3a = _read_csv(p3a_csv)
        if all(c in p3a.columns for c in ["chrom", "pos_based", "cell_type"]):
            df = df.merge(p3a, on=["chrom", "pos_based", "cell_type"], how="left")
        elif all(c in p3a.columns for c in ["chrom", "pos_based"]):
            df = df.merge(p3a, on=["chrom", "pos_based"], how="left")
        else:
            raise ValueError("P3a output must contain chrom,pos_based (and preferably cell_type).")
        print(f"After merge P3a dynamic: {df.shape}")

    # ---------- merge P3b promoter mC ----------
    if args.use_p3b_promoter_mC:
        if not os.path.exists(args.p3b_promoter_mc_csv):
            raise FileNotFoundError(f"P3b promoter mC not found: {args.p3b_promoter_mc_csv}")
        prom = _read_csv(args.p3b_promoter_mc_csv)
        _ensure_cols(prom, ["gene_id"], "p3b_promoter_mc")

        missing_mc = [c for c in MC_COLS if c not in prom.columns]
        if missing_mc:
            raise ValueError(f"P3b promoter mC file missing: {missing_mc}")

        if "gene_name" in prom.columns and "gene_name" in df.columns:
            prom = prom.drop(columns=["gene_name"])

        df["gene_id_base"] = df["gene_id"].astype(str).map(norm_gid)
        prom["gene_id_base"] = prom["gene_id"].astype(str).map(norm_gid)
        prom = prom.drop(columns=["gene_id"]).drop_duplicates(subset=["gene_id_base"])

        df = df.merge(prom, on="gene_id_base", how="left")
        print(f"After merge P3b promoter mC: {df.shape}")

        if args.add_relative_mC_features:
            df = _add_promoter_relative_features(df)
            print("Added promoter relative features: this_cell/mean/std/delta/z/rank")

    # ---------- relative features ----------
    if args.add_relative_mC_features:
        df = _add_site_relative_features_from_p3a(df)
        if "mC_site_delta" in df.columns:
            print("Added site relative features from P3a: delta/z/range_over_mean/top2gap_over_std")

        df = _add_site_promoter_coupling_features(df)
        couple_cols = [c for c in ["mC_site_minus_promoter", "mC_site_over_promoter", "mC_site_z_minus_promoter_z"] if c in df.columns]
        if couple_cols:
            print("Added site-promoter coupling features:", couple_cols)

    if args.add_stage2_interactions:
        before_cols = set(df.columns)
        df = _add_stage2_interaction_features(df)
        added_cols = [c for c in df.columns if c not in before_cols]
        print(f"Added Stage2 interaction features: {added_cols}")

    # ---------- cleanup leakage ----------
    leak_cols = [
        c for c in df.columns if c in [
            "5hmc_signal",           # direct label source
            "rank_pct_cell",         # row-level 5hmc-derived
            "zscore_cell",           # row-level 5hmc-derived
            "thr_cell_used",         # label construction artifact
            "label_confidence",      # converted to sample_weight if requested
            "label_cell_mode",       # constant / config marker
            "label_site", "site_role", "pos_count", "obs_count", "pos_ratio"
        ]
    ]
    if leak_cols:
        df = df.drop(columns=leak_cols)
        print(f"Dropped leakage columns: {leak_cols}")

    # gene_id_base no longer needed
    if "gene_id_base" in df.columns:
        df = df.drop(columns=["gene_id_base"])

    # ---------- save ----------
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, na_rep="NaN")
    print("=" * 100)
    print("✅ Saved Stage-2 dataset:", out_csv)
    print("Shape:", df.shape)
    if "sample_weight" in df.columns:
        sw = pd.to_numeric(df["sample_weight"], errors="coerce")
        print("sample_weight summary:", {
            "min": float(sw.min()),
            "p25": float(sw.quantile(0.25)),
            "median": float(sw.quantile(0.5)),
            "p75": float(sw.quantile(0.75)),
            "max": float(sw.max()),
        })
    print("=" * 100)


if __name__ == "__main__":
    main()
