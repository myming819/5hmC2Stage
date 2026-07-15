#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np


def detect_cell_types(df: pd.DataFrame):
    hmc_cols = [c for c in df.columns if c.startswith("5hmc_signal_")]
    if len(hmc_cols) == 0:
        raise ValueError("No columns like 5hmc_signal_<celltype> found.")
    return [c.replace("5hmc_signal_", "", 1) for c in hmc_cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--meta_csv", default="",
                    help="optional stage2 csv to merge sample_weight by chrom,pos_based,cell_type")
    ap.add_argument("--meta_weight_col", default="sample_weight")
    ap.add_argument("--dropna_hmc", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    if "chrom" not in df.columns or "pos_based" not in df.columns:
        raise ValueError("input_csv must contain chrom and pos_based")

    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["pos_based"] = pd.to_numeric(df["pos_based"], errors="coerce")
    df = df[df["pos_based"].notna()].copy()
    df["pos_based"] = df["pos_based"].astype(int)

    cell_types = detect_cell_types(df)

    rows = []
    for ct in cell_types:
        hmc_col = f"5hmc_signal_{ct}"
        mc_col = f"5mc_signal_{ct}"

        part = pd.DataFrame({
            "chrom": df["chrom"],
            "pos_based": df["pos_based"],
            "cell_type": ct,
            "hmc_signal_cont": pd.to_numeric(df[hmc_col], errors="coerce"),
        })

        if mc_col in df.columns:
            part["mc_signal"] = pd.to_numeric(df[mc_col], errors="coerce")
        else:
            part["mc_signal"] = np.nan

        part["sample_weight"] = 1.0
        rows.append(part)

    long_df = pd.concat(rows, axis=0, ignore_index=True)

    if args.meta_csv:
        meta = pd.read_csv(args.meta_csv, low_memory=False)
        need = ["chrom", "pos_based", "cell_type", args.meta_weight_col]
        miss = [c for c in need if c not in meta.columns]
        if miss:
            raise ValueError(f"meta_csv missing columns: {miss}")

        meta["chrom"] = meta["chrom"].astype(str).str.strip()
        meta["pos_based"] = pd.to_numeric(meta["pos_based"], errors="coerce")
        meta = meta[meta["pos_based"].notna()].copy()
        meta["pos_based"] = meta["pos_based"].astype(int)
        meta["cell_type"] = meta["cell_type"].astype(str)
        meta[args.meta_weight_col] = pd.to_numeric(meta[args.meta_weight_col], errors="coerce")

        meta = (
            meta[["chrom", "pos_based", "cell_type", args.meta_weight_col]]
            .drop_duplicates(["chrom", "pos_based", "cell_type"])
            .rename(columns={args.meta_weight_col: "sample_weight"})
        )

        long_df = long_df.drop(columns=["sample_weight"], errors="ignore").merge(
            meta,
            on=["chrom", "pos_based", "cell_type"],
            how="left"
        )
        long_df["sample_weight"] = long_df["sample_weight"].fillna(1.0)

    if args.dropna_hmc:
        long_df = long_df[long_df["hmc_signal_cont"].notna()].copy()

    long_df.to_csv(args.out_csv, index=False)

    print("=" * 80)
    print("✅ continuous long table exported")
    print("input :", args.input_csv)
    print("output:", args.out_csv)
    print("shape :", long_df.shape)
    print("cell types:", sorted(long_df["cell_type"].unique().tolist()))
    print("=" * 80)


if __name__ == "__main__":
    main()