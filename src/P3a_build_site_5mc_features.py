#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P3a_dynamic_site_mC.py
Stage3a: Build dynamic site-level mC features.

Input:
  P1 long table:
    data/data_out/Brain_CPG_ALL_UNION/ALL_cells_{union|intersection}_CpG_long_table.csv
  (needs: chrom, pos_based, cell_type, 5mc_signal)

Output:
  feature_output/Brain_CPG_ALL_UNION/dynamic_site_mC_from_{mode}.csv
  columns include:
    chrom, pos_based, cell_type,
    mC_site_this_cell (=5mc_signal),
    mC_site_mean/std/min/max/range/non_nan_count/top2gap (site-aggregate across cells)
"""

import os
import argparse
import numpy as np
import pandas as pd


def resolve_long_csv(base_dir: str, mode: str, long_csv: str) -> str:
    if long_csv and long_csv.strip():
        return long_csv.strip()
    mode = mode.lower().strip()
    return os.path.join(base_dir, f"ALL_cells_{mode}_CpG_long_table.csv")


def default_out_csv(out_dir: str, mode: str) -> str:
    mode = mode.lower().strip()
    return os.path.join(out_dir, f"dynamic_site_mC_from_{mode}.csv")


def compute_site_mc_stats(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-site aggregate stats across all cell rows:
      mean/std/min/max/range/non_nan_count/top2gap
    """
    work = df_long[["chrom", "pos_based", "5mc_signal"]].copy()
    work["chrom"] = work["chrom"].astype(str)
    work["pos_based"] = pd.to_numeric(work["pos_based"], errors="coerce").astype("Int64")
    work["5mc_signal"] = pd.to_numeric(work["5mc_signal"], errors="coerce").astype("float32")
    work = work.dropna(subset=["pos_based"]).copy()

    # group aggregate (use standard groupby for compatibility)
    g = work.groupby(["chrom", "pos_based"])["5mc_signal"]
    stats = g.agg(
        mC_site_mean="mean",
        mC_site_std="std",
        mC_site_min="min",
        mC_site_max="max",
        mC_site_non_nan_count="count"
    ).reset_index()

    stats["mC_site_range"] = stats["mC_site_max"] - stats["mC_site_min"]

    # top2gap (stable across pandas versions)
    def top2gap(s):
        x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
        if x.size < 2:
            return np.nan
        x.sort()
        return float(x[-1] - x[-2])

    # IMPORTANT: use groupby WITHOUT as_index=False for stable output
    g2 = work.groupby(["chrom", "pos_based"])["5mc_signal"]

    top2 = g2.agg(mC_site_top2gap=top2gap).reset_index()

    # merge
    out = stats.merge(top2, on=["chrom", "pos_based"], how="left")

    # fill std NaN (single obs) -> 0
    out["mC_site_std"] = out["mC_site_std"].fillna(0.0)

    # enforce float32
    for c in ["mC_site_mean","mC_site_std","mC_site_min","mC_site_max","mC_site_range","mC_site_top2gap"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    out["mC_site_non_nan_count"] = pd.to_numeric(out["mC_site_non_nan_count"], errors="coerce").astype("int32")
    return out


def main():
    ap = argparse.ArgumentParser(description="P3a: Dynamic site mC features from P1 long table")
    ap.add_argument("--base_dir", default=r"data/data_out/PBMC_CPG_ALL_UNION")
    ap.add_argument("--input_mode", choices=["union","intersection"], default="union")
    ap.add_argument("--long_csv", default="", help="optional manual P1 long table path")

    ap.add_argument("--out_dir", default=r"feature_output/PBMC_CPG_ALL_UNION")
    ap.add_argument("--out_csv", default="", help="optional output csv path")

    ap.add_argument("--no_site_agg", action="store_true",
                    help="if set, only output mC_site_this_cell (=5mc_signal) without site aggregates")

    args = ap.parse_args()

    long_csv = resolve_long_csv(args.base_dir, args.input_mode, args.long_csv)
    if not os.path.exists(long_csv):
        raise FileNotFoundError(f"Long table not found: {long_csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = args.out_csv.strip() if args.out_csv.strip() else default_out_csv(args.out_dir, args.input_mode)

    df = pd.read_csv(long_csv, low_memory=False)
    need = ["chrom","pos_based","cell_type","5mc_signal"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"P1 long table missing columns: {miss}")

    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["cell_type"] = df["cell_type"].astype(str).str.strip()
    df["pos_based"] = pd.to_numeric(df["pos_based"], errors="coerce").astype("Int64")
    df["5mc_signal"] = pd.to_numeric(df["5mc_signal"], errors="coerce").astype("float32")
    df = df.dropna(subset=["pos_based"]).copy()

    out = df[["chrom","pos_based","cell_type"]].copy()
    out["mC_site_this_cell"] = df["5mc_signal"].copy()

    if not args.no_site_agg:
        stats = compute_site_mc_stats(df)
        out = out.merge(stats, on=["chrom","pos_based"], how="left")

    out.to_csv(out_csv, index=False, na_rep="NaN")
    print("="*90)
    print("✅ P3a done: dynamic site mC features")
    print(f"Input : {long_csv}")
    print(f"Output: {out_csv}")
    print(f"Rows  : {len(out):,} | Cols: {len(out.columns)}")
    print("="*90)


if __name__ == "__main__":
    main()