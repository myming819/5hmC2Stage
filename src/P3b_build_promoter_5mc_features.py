#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P3b_dynamic_promoter_mC.py
Stage3b: Compute gene-promoter 5mC (gene×cell) from variableStep WIG + promoter BED.

Inputs:
  - promoter BED: chrom start end gene_id gene_name strand (0-based BED)
  - either:
      (A) --use_default_wig_map : use built-in WIG_FILES_5MC_DEFAULT
      (B) --wig_map_csv         : csv with columns mc_col, wig_path

Output:
  - out_csv: gene_id,gene_name, mC_OPC,mC_ODC1,... (gene×cell)
"""

import os
import re
import argparse
import numpy as np
import pandas as pd

# ------------------ default Brain 11-celltype 5mC WIG paths ------------------
WIG_FILES_5MC_DEFAULT = {
    "mC_OPC":  r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_OPC.bs_1.wig\Brain_5mC_CPG_1_OPC.bs_1.wig",
    "mC_ODC1": r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_ODC1.bs_1.wig\Brain_5mC_CPG_1_ODC1.bs_1.wig",
    "mC_ODC2": r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_ODC2.bs_1.wig\Brain_5mC_CPG_1_ODC2.bs_1.wig",
    "mC_ODC3": r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_ODC3.bs_1.wig\Brain_5mC_CPG_1_ODC3.bs_1.wig",
    "mC_MGC":  r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_MGC.bs_1.wig\Brain_5mC_CPG_1_MGC.bs_1.wig",
    "mC_INH":  r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_INH.bs_1.wig\Brain_5mC_CPG_1_INH.bs_1.wig",
    "mC_ENDO": r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_ENDO.bs_1.wig\Brain_5mC_CPG_1_ENDO.bs_1.wig",
    "mC_ASC1": r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_ASC1.bs_1.wig\Brain_5mC_CPG_1_ASC1.bs_1.wig",
    "mC_ASC2": r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_ASC2.bs_1.wig\Brain_5mC_CPG_1_ASC2.bs_1.wig",
    "mC_EXC1": r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_EXC1.bs_1.wig\Brain_5mC_CPG_1_EXC1.bs_1.wig",
    "mC_EXC2": r"data\GSE197740_RAW\GSM7291623_Brain_wig\Brain_wig\Brain_5mC_CPG_1_EXC2.bs_1.wig\Brain_5mC_CPG_1_EXC2.bs_1.wig",
}

# WIG_FILES_5MC_DEFAULT  = {
#     "mC_B":  r"data\GSE197740_RAW\GSM5929313_PBMC_wig\PBMC_wig\PBMC_5mC_CPG_1_B.bs_1.wig\PBMC_5mC_CPG_1_B.bs_1.wig",
#     "mC_T_reg": r"data\GSE197740_RAW\GSM5929313_PBMC_wig\PBMC_wig\PBMC_5mC_CPG_1_T_reg.bs_1.wig\PBMC_5mC_CPG_1_T_reg.bs_1.wig",
#     "mC_T_naive": r"data\GSE197740_RAW\GSM5929313_PBMC_wig\PBMC_wig\PBMC_5mC_CPG_1_T_naive.bs_1.wig\PBMC_5mC_CPG_1_T_naive.bs_1.wig",
#     "mC_NK": r"data\GSE197740_RAW\GSM5929313_PBMC_wig\PBMC_wig\PBMC_5mC_CPG_1_NK.bs_1.wig\PBMC_5mC_CPG_1_NK.bs_1.wig",
#     "mC_Monocytes":  r"data\GSE197740_RAW\GSM5929313_PBMC_wig\PBMC_wig\PBMC_5mC_CPG_1_Monocytes.bs_1.wig\PBMC_5mC_CPG_1_Monocytes.bs_1.wig"
# }

# WIG_FILES_5MC_DEFAULT = {
#     "mC_2i":  r"data\GSE197740_RAW\GSM5929312_mESC_wig\mESC_wig\mESC_5mC_CPG_1_2i.bs_1.wig\mESC_5mC_CPG_1_2i.bs_1.wig",
#     "mC_serum": r"data\GSE197740_RAW\GSM5929312_mESC_wig\mESC_wig\mESC_5mC_CPG_1_serum.bs_1.wig\mESC_5mC_CPG_1_serum.bs_1.wig"
# }
def load_promoter_bed(promoter_bed_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        promoter_bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "gene_id", "gene_name", "strand"],
        low_memory=False
    )
    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    df["gene_name"] = df["gene_name"].astype(str).str.strip()
    df = df.dropna(subset=["chrom", "start", "end", "gene_id", "gene_name"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def normalize_chrom_name(chrom: str, mode: str) -> str:
    """
    mode:
      - keep: no change
      - add_chr: ensure starts with 'chr'
      - drop_chr: ensure does NOT start with 'chr'
    """
    c = str(chrom).strip()
    if mode == "keep":
        return c
    if mode == "add_chr":
        return c if c.startswith("chr") else f"chr{c}"
    if mode == "drop_chr":
        return c[3:] if c.startswith("chr") else c
    raise ValueError("chrom_prefix_mode must be keep/add_chr/drop_chr")


def load_wig_as_dict(wig_path: str, chrom_prefix_mode: str = "keep"):
    """
    Parse variableStep WIG.
    Return dict: chrom -> (pos_arr, val_arr)
      pos_arr: 1-based positions sorted
      val_arr: float32 values aligned to pos_arr
    """
    chrom_data = {}
    current_chrom = None

    with open(wig_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("track"):
                continue
            if line.startswith("variableStep"):
                m = re.search(r"chrom=([^\s]+)", line)
                current_chrom = m.group(1) if m else None
                if current_chrom is None:
                    continue
                current_chrom = normalize_chrom_name(current_chrom, chrom_prefix_mode)
                if current_chrom not in chrom_data:
                    chrom_data[current_chrom] = {"pos": [], "val": []}
                continue
            parts = line.split()
            if len(parts) != 2 or current_chrom is None:
                continue
            pos = int(parts[0])     # 1-based
            val = float(parts[1])
            chrom_data[current_chrom]["pos"].append(pos)
            chrom_data[current_chrom]["val"].append(val)

    out = {}
    total_points = 0
    for chrom, d in chrom_data.items():
        pos_arr = np.array(d["pos"], dtype=np.int64)
        val_arr = np.array(d["val"], dtype=np.float32)
        if pos_arr.size == 0:
            continue
        order = np.argsort(pos_arr)
        out[chrom] = (pos_arr[order], val_arr[order])
        total_points += pos_arr.size
    return out, total_points


def interval_mean_from_wig(chrom_to_data: dict, chrom: str, start0: int, end0: int, chrom_prefix_mode: str) -> float:
    """
    Mean wig value within [start0, end0) in 0-based coordinates.
    WIG positions are 1-based.
    """
    chrom = normalize_chrom_name(chrom, chrom_prefix_mode)
    if chrom not in chrom_to_data:
        return np.nan
    pos_arr, val_arr = chrom_to_data[chrom]
    start1 = int(start0) + 1
    end1 = int(end0)  # exclusive boundary
    l = np.searchsorted(pos_arr, start1, side="left")
    r = np.searchsorted(pos_arr, end1, side="right")
    if r <= l:
        return np.nan
    return float(np.mean(val_arr[l:r]))


def compute_promoter_mc(promoter_df: pd.DataFrame, chrom_to_data: dict, chrom_prefix_mode: str) -> np.ndarray:
    out = np.empty(len(promoter_df), dtype=np.float32)
    for i, row in enumerate(promoter_df.itertuples(index=False)):
        out[i] = interval_mean_from_wig(chrom_to_data, row.chrom, row.start, row.end, chrom_prefix_mode)
    return out


def read_wig_map(wig_map_csv: str) -> pd.DataFrame:
    df = pd.read_csv(wig_map_csv, low_memory=False)
    cols = [c.lower() for c in df.columns]
    if "mc_col" not in cols or "wig_path" not in cols:
        raise ValueError("wig_map_csv must contain columns: mc_col, wig_path")
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)
    df["mc_col"] = df["mc_col"].astype(str).str.strip()
    df["wig_path"] = df["wig_path"].astype(str).str.strip()
    df = df[df["mc_col"].ne("") & df["wig_path"].ne("")].copy()
    return df


def main():
    ap = argparse.ArgumentParser(description="P3b: gene-promoter 5mC from WIG + promoter BED")

    ap.add_argument("--promoter_bed", default=r"data\human_hg38_promoter_500bp.bed",
                    help="promoter BED (chrom,start,end,gene_id,gene_name,strand)")

    ap.add_argument("--wig_map_csv", default="",
                    help="optional csv with columns: mc_col, wig_path")

    ap.add_argument("--use_default_wig_map", action="store_true",
                    help="use built-in WIG_FILES_5MC_DEFAULT")

    ap.add_argument("--out_csv", default=r"data\data_out\PBMC_CPG_ALL_UNION\PBMC_gene_promoter_5mC_11celltypes.csv",
                    help="output gene×cell promoter 5mC csv")

    ap.add_argument("--chrom_prefix_mode", choices=["keep", "add_chr", "drop_chr"], default="keep",
                    help="resolve chr prefix mismatch between WIG and BED")

    ap.add_argument("--limit_genes", type=int, default=0,
                    help="debug: if >0, only compute first N promoters")

    args = ap.parse_args()

    if not os.path.exists(args.promoter_bed):
        raise FileNotFoundError(f"promoter_bed not found: {args.promoter_bed}")

    # IMPORTANT: only check wig_map_csv existence when not using default mapping
    if (not args.use_default_wig_map) and args.wig_map_csv.strip():
        if not os.path.exists(args.wig_map_csv):
            raise FileNotFoundError(f"wig_map_csv not found: {args.wig_map_csv}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    print("=" * 90)
    print("🧬 P3b: Compute gene-promoter 5mC (gene×cell)")
    print("=" * 90)
    print(f"Promoter BED      : {args.promoter_bed}")
    print(f"Use default wigmap: {args.use_default_wig_map}")
    print(f"WIG map csv       : {args.wig_map_csv if args.wig_map_csv else '(none)'}")
    print(f"Chrom prefix mode : {args.chrom_prefix_mode}")
    print(f"Output            : {args.out_csv}")

    promoter_df = load_promoter_bed(args.promoter_bed)
    if args.limit_genes and args.limit_genes > 0:
        promoter_df = promoter_df.iloc[: int(args.limit_genes)].copy()
        print(f"⚠️ limit_genes enabled: using first {len(promoter_df)} promoters")

    out = promoter_df[["gene_id", "gene_name"]].copy()

    if args.use_default_wig_map:
        items = list(WIG_FILES_5MC_DEFAULT.items())
    else:
        if not args.wig_map_csv.strip():
            raise ValueError("Please provide --wig_map_csv or use --use_default_wig_map")
        wig_map = read_wig_map(args.wig_map_csv)
        items = list(zip(wig_map["mc_col"], wig_map["wig_path"]))

    for mc_col, wig_path in items:
        if not os.path.exists(wig_path):
            raise FileNotFoundError(f"WIG not found: {wig_path}")

        print(f"\n[WIG] Loading {mc_col}")
        chrom_to_data, n_points = load_wig_as_dict(wig_path, chrom_prefix_mode=args.chrom_prefix_mode)
        print(f"   - chroms loaded: {len(chrom_to_data)} | points: {n_points:,}")

        print("   - computing promoter mean ...")
        out[mc_col] = compute_promoter_mc(promoter_df, chrom_to_data, chrom_prefix_mode=args.chrom_prefix_mode)

        nn = int(pd.to_numeric(out[mc_col], errors="coerce").notna().sum())
        print(f"   - {mc_col} non-NaN promoters: {nn:,}/{len(out):,}")

    out.to_csv(args.out_csv, index=False, na_rep="NaN")
    print("\n✅ Saved:", args.out_csv)
    print("Shape:", out.shape)
    print("=" * 90)


if __name__ == "__main__":
    main()