#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P4_build_stage1_dataset.py
Build Stage-1 site-level STATIC dataset:
  label = label_site
  features = seq_features + (optional) dist_to_TSS + (optional) gene TF motif scores

Inputs:
  - P1 Stage1 site dataset: ALL_cells_{mode}_Stage1_site_dataset.csv
    (pos/neg only, already downsampled)
  - P2a seq features: feature_output/.../seq_features_from_{mode}.csv
  - (optional) P2b long_with_gene: ALL_cells_{mode}_CpG_long_table_with_gene.csv
    (for mapping site->gene_id/gene_name/dist_to_TSS)
  - (optional) P2c gene TF list: feature_output/.../gene_TFlist_maxscore_topK{K}.csv

Output:
  - out_csv (default): model_input/Brain_CPG_ALL_UNION/stage1_site_static_{mode}.csv
"""

import os
import argparse
import pandas as pd
import numpy as np


def norm_gid(g: str) -> str:
    g = str(g).strip()
    if g in {"", "nan", "None"}:
        return ""
    return g.split(".")[0]


def resolve_p1_stage1_sites(base_dir: str, mode: str, path: str) -> str:
    if path and path.strip():
        return path.strip()
    return os.path.join(base_dir, f"ALL_cells_{mode}_Stage1_site_dataset.csv")


def resolve_p2a_seq(out_dir: str, mode: str, path: str) -> str:
    if path and path.strip():
        return path.strip()
    return os.path.join(out_dir, f"seq_features_from_{mode}.csv")


def resolve_p2b_long_with_gene(base_dir: str, mode: str, path: str) -> str:
    if path and path.strip():
        return path.strip()
    return os.path.join(base_dir, f"ALL_cells_{mode}_CpG_long_table_with_gene.csv")


def resolve_p2c_tf(tf_dir: str, topk: int, path: str) -> str:
    if path and path.strip():
        return path.strip()
    return os.path.join(tf_dir, f"gene_TFlist_maxscore_topK{int(topk)}.csv")


def default_out(out_dir: str, mode: str) -> str:
    return os.path.join(out_dir, f"stage1_site_static_{mode}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=r"data/data_out/Brain_CPG_ALL_UNION")
    ap.add_argument("--input_mode", choices=["union", "intersection"], default="union")

    ap.add_argument("--seq_dir", default=r"feature_output/Brain_CPG_ALL_UNION")
    ap.add_argument("--tf_dir", default=r"feature_output/Brain_CPG_ALL_UNION")
    ap.add_argument("--out_dir", default=r"model_input/Brain_CPG_ALL_UNION")

    ap.add_argument("--p1_stage1_sites", default="")
    ap.add_argument("--p2a_seq_csv", default="")
    ap.add_argument("--p2b_long_with_gene", default="")
    ap.add_argument("--p2c_tf_csv", default="")
    ap.add_argument("--tf_topk", type=int, default=100)

    ap.add_argument("--use_gene_mapping", action="store_true",
                    help="Include gene_id/gene_name/dist_to_TSS by mapping from P2b long_with_gene")
    ap.add_argument("--use_tf", action="store_true",
                    help="Include TF motif features by gene_id join (requires --use_gene_mapping)")
    ap.add_argument("--out_csv", default="")

    args = ap.parse_args()
    mode = args.input_mode.lower()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = args.out_csv.strip() if args.out_csv.strip() else default_out(args.out_dir, mode)

    p1_sites = resolve_p1_stage1_sites(args.base_dir, mode, args.p1_stage1_sites)
    p2a_seq = resolve_p2a_seq(args.seq_dir, mode, args.p2a_seq_csv)

    if not os.path.exists(p1_sites):
        raise FileNotFoundError(p1_sites)
    if not os.path.exists(p2a_seq):
        raise FileNotFoundError(p2a_seq)

    print("=" * 90)
    print("📦 P4 Stage-1 dataset builder (STATIC site-level)")
    print("=" * 90)
    print("mode:", mode)
    print("P1 Stage1 sites:", p1_sites)
    print("P2a seq features:", p2a_seq)
    print("use_gene_mapping:", args.use_gene_mapping, "| use_tf:", args.use_tf)
    print("Output:", out_csv)

    # 1) load Stage1 site list (pos/neg only)
    sites = pd.read_csv(p1_sites, low_memory=False)
    need = ["chrom", "pos_based", "label_site"]
    miss = [c for c in need if c not in sites.columns]
    if miss:
        raise ValueError(f"P1 Stage1 sites missing columns: {miss}")
    sites["chrom"] = sites["chrom"].astype(str).str.strip()
    sites["pos_based"] = pd.to_numeric(sites["pos_based"], errors="coerce").astype("Int64")
    sites = sites.dropna(subset=["pos_based"]).copy()
    sites["pos_based"] = sites["pos_based"].astype(int)

    # 2) load seq features
    seq = pd.read_csv(p2a_seq, low_memory=False)
    if not {"chrom", "pos_based"}.issubset(seq.columns):
        raise ValueError("P2a seq features must contain chrom,pos_based")
    seq["chrom"] = seq["chrom"].astype(str).str.strip()
    seq["pos_based"] = pd.to_numeric(seq["pos_based"], errors="coerce").astype("Int64")
    seq = seq.dropna(subset=["pos_based"]).copy()
    seq["pos_based"] = seq["pos_based"].astype(int)

    # 3) merge Stage1 sites with seq features
    df = sites.merge(seq, on=["chrom", "pos_based"], how="left")
    n_missing_seq = int(df.filter(regex=r"^BIO_|^POS_|^kmer").isna().all(axis=1).sum())
    print(f"After merge seq: rows={len(df):,} | missing_seq_rows≈{n_missing_seq:,}")

    # 4) optional gene mapping (site -> gene_id/gene_name/dist_to_TSS) from P2b long_with_gene
    if args.use_gene_mapping:
        p2b = resolve_p2b_long_with_gene(args.base_dir, mode, args.p2b_long_with_gene)
        if not os.path.exists(p2b):
            raise FileNotFoundError(p2b)
        lwg = pd.read_csv(p2b, low_memory=False, usecols=[c for c in ["chrom","pos_based","gene_id","gene_name","dist_to_TSS"] if True])
        lwg["chrom"] = lwg["chrom"].astype(str).str.strip()
        lwg["pos_based"] = pd.to_numeric(lwg["pos_based"], errors="coerce").astype("Int64")
        lwg = lwg.dropna(subset=["pos_based"]).copy()
        lwg["pos_based"] = lwg["pos_based"].astype(int)
        # unique mapping at site-level
        map_site = lwg.drop_duplicates(subset=["chrom","pos_based"])[["chrom","pos_based","gene_id","gene_name","dist_to_TSS"]]
        df = df.merge(map_site, on=["chrom","pos_based"], how="left")
        print("After merge gene mapping: cols=", len(df.columns))

        # 5) optional TF features (gene-level)
        if args.use_tf:
            if not args.use_gene_mapping:
                raise ValueError("--use_tf requires --use_gene_mapping")
            tf_csv = resolve_p2c_tf(args.tf_dir, args.tf_topk, args.p2c_tf_csv)
            if not os.path.exists(tf_csv):
                raise FileNotFoundError(tf_csv)
            tf = pd.read_csv(tf_csv, low_memory=False)
            if "gene_id" not in tf.columns:
                raise ValueError("TF csv must have gene_id column")
            tf["gene_id_base"] = tf["gene_id"].astype(str).map(norm_gid)
            df["gene_id_base"] = df["gene_id"].astype(str).map(norm_gid)
            tf = tf.drop(columns=["gene_id"]).rename(columns={"gene_id_base": "gene_id_base"})
            # merge
            df = df.merge(tf, on="gene_id_base", how="left")
            df = df.drop(columns=["gene_id_base"])
            print("After merge TF: cols=", len(df.columns))

    # 6) sanity: ensure no dynamic columns
    dyn_like = [c for c in df.columns if ("mC_" in c) or (c in {"5mc_signal", "5hmc_signal", "thr_cell_gmm", "label_cell"})]
    if dyn_like:
        print("⚠️ Warning: dynamic-like columns found in Stage1 dataset (should NOT):", dyn_like[:20])

    # 7) save
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False, na_rep="NaN")
    print("=" * 90)
    print("✅ Saved Stage-1 dataset:", out_csv)
    print("Shape:", df.shape)
    print("=" * 90)


if __name__ == "__main__":
    main()