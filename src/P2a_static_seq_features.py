#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P2a_static_seq_features.py
Stage2a: Extract CpG-site-level STATIC sequence features (BIO_/POS_/kmer) from genome FASTA.

Input:
  P1 long table:
    data/data_out/Brain_CPG_ALL_UNION/ALL_cells_{union|intersection}_CpG_long_table.csv
  (needs chrom, pos_based; we deduplicate to site-level)

Output:
  feature_output/Brain_CPG_ALL_UNION/seq_features_from_{mode}.csv
  columns: chrom, pos_based, BIO_*, POS_*, kmer3_*, (optional kmer2_*, kmer3c_*)

This is a pipeline-friendly refactor of your A2_feature_creat_seq_feature.py
(keeps its core behavior: dedup sites -> extract flanking sequence -> compute features).
"""

import os
import argparse
import logging
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NUCLEOTIDES_DNA = ["A", "C", "G", "T"]
NUC2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
N = -1


# ------------------ FASTA parsing ------------------
def parse_genome_fasta(fasta_path: str) -> Dict[str, str]:
    """
    Simple FASTA loader: returns dict chrom -> sequence (upper).
    Works for standard genomes; assumes memory is ok.
    """
    chr_seq = {}
    cur = None
    parts = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    chr_seq[cur] = "".join(parts).upper()
                cur = line[1:].strip().split()[0]
                parts = []
            else:
                parts.append(line.strip())
        if cur is not None:
            chr_seq[cur] = "".join(parts).upper()
    return chr_seq


def get_flanking_seq(chr_seq_dict: Dict[str, str], chrom: str, pos_1based: int, window_radius: int) -> str:
    """
    Extract window centered at pos_1based (1-based), length = 2r+1.
    Pads with 'N' if out of boundary or chrom missing.
    """
    r = int(window_radius)
    L = 2 * r + 1
    if chrom not in chr_seq_dict:
        return "N" * L

    seq = chr_seq_dict[chrom]
    n = len(seq)
    center0 = int(pos_1based) - 1  # 0-based center index
    left0 = center0 - r
    right0 = center0 + r

    # if fully in range
    if left0 >= 0 and right0 < n:
        return seq[left0:right0 + 1]

    # boundary padding
    out = []
    for i in range(left0, right0 + 1):
        if 0 <= i < n:
            out.append(seq[i])
        else:
            out.append("N")
    return "".join(out)


# ------------------ Sequence encoding ------------------
def seq_to_numeric(seq: str) -> np.ndarray:
    """
    Encode A,C,G,T -> 0,1,2,3; others -> -1
    """
    seq = (seq or "").upper()
    arr = np.empty(len(seq), dtype=np.int8)
    for i, ch in enumerate(seq):
        arr[i] = NUC2IDX.get(ch, N)
    return arr


# ------------------ Feature blocks ------------------
def bio_feature_names():
    return [
        "BIO_valid_len",
        "BIO_pA", "BIO_pC", "BIO_pG", "BIO_pT",
        "BIO_GC", "BIO_AT",
        "BIO_GC_skew", "BIO_AT_skew",
        "BIO_CpG_count", "BIO_CHG_count", "BIO_CHH_count",
        "BIO_CpG_density", "BIO_CHG_density", "BIO_CHH_density",
        "BIO_CpG_OE",
        "BIO_CpG_gap_mean", "BIO_CpG_gap_std",
    ]


def extract_bio_features(seq_num: np.ndarray) -> np.ndarray:
    """
    Compute composition & CpG/CHG/CHH summary on the whole window.
    """
    n = len(seq_num)
    valid = (seq_num >= 0)
    valid_len = int(valid.sum())
    if valid_len == 0:
        return np.zeros(len(bio_feature_names()), dtype=np.float32)

    counts = np.bincount(seq_num[valid], minlength=4).astype(np.float32)
    pA, pC, pG, pT = counts / float(valid_len)
    gc = float(pG + pC)
    at = float(pA + pT)
    gc_skew = float((pG - pC) / (pG + pC + 1e-8))
    at_skew = float((pA - pT) / (pA + pT + 1e-8))

    # dinucleotide/trinucleotide patterns around C
    # CpG: C followed by G
    # CHG: C (not G) ? actually "H" = not G; CHG means C, not G, then G
    # CHH: C, not G, then not G
    cpg = 0
    chg = 0
    chh = 0
    c_positions = np.where(seq_num[:-1] == 1)[0]  # positions of 'C' except last base
    for i in c_positions:
        if i + 1 < n and seq_num[i + 1] == 2:  # G
            cpg += 1
        if i + 2 < n:
            b2 = seq_num[i + 1]
            b3 = seq_num[i + 2]
            if b2 >= 0 and b3 >= 0:
                if b2 != 2 and b3 == 2:
                    chg += 1
                if b2 != 2 and b3 != 2:
                    chh += 1

    # densities per valid length
    cpg_density = float(cpg / (valid_len + 1e-8))
    chg_density = float(chg / (valid_len + 1e-8))
    chh_density = float(chh / (valid_len + 1e-8))

    # CpG observed/expected
    pC = float(pC)
    pG = float(pG)
    cpg_oe = float((cpg_density + 1e-8) / (pC * pG + 1e-8))

    # CpG gaps: distances between CpG starts (approx)
    cpg_starts = []
    for i in range(n - 1):
        if seq_num[i] == 1 and seq_num[i + 1] == 2:
            cpg_starts.append(i)
    if len(cpg_starts) >= 2:
        gaps = np.diff(np.array(cpg_starts, dtype=np.int32)).astype(np.float32)
        gap_mean = float(gaps.mean())
        gap_std = float(gaps.std())
    else:
        gap_mean, gap_std = 0.0, 0.0

    feats = np.array([
        float(valid_len),
        float(pA), float(pC), float(pG), float(pT),
        float(gc), float(at),
        float(gc_skew), float(at_skew),
        float(cpg), float(chg), float(chh),
        float(cpg_density), float(chg_density), float(chh_density),
        float(cpg_oe),
        float(gap_mean), float(gap_std)
    ], dtype=np.float32)
    return feats


def pos_feature_names(radius: int):
    names = []
    r = int(radius)
    # positions: -r..-1, +1..+r (exclude center to avoid trivial CpG 'C' itself)
    for off in list(range(-r, 0)) + list(range(1, r + 1)):
        for b in NUCLEOTIDES_DNA:
            names.append(f"POS_off{off}_{b}")
    return names


def extract_positional_onehot(seq_num: np.ndarray, radius: int) -> np.ndarray:
    """
    One-hot around the center (exclude center), shape = (2r*4)
    """
    r = int(radius)
    n = len(seq_num)
    center = n // 2
    vec = np.zeros((2 * r) * 4, dtype=np.float32)
    idx = 0
    for off in list(range(-r, 0)) + list(range(1, r + 1)):
        pos = center + off
        b = int(seq_num[pos]) if 0 <= pos < n else N
        for k in range(4):
            vec[idx + k] = 1.0 if b == k else 0.0
        idx += 4
    return vec


MER2_NAMES = [a + b for a in NUCLEOTIDES_DNA for b in NUCLEOTIDES_DNA]
MER2_MAP = {k: i for i, k in enumerate(MER2_NAMES)}
MER3_NAMES = [a + b + c for a in NUCLEOTIDES_DNA for b in NUCLEOTIDES_DNA for c in NUCLEOTIDES_DNA]
MER3_MAP = {k: i for i, k in enumerate(MER3_NAMES)}


def kmer2_feature_names():
    return [f"kmer2_{d}" for d in MER2_NAMES]


def kmer3_feature_names(prefix: str):
    return [f"{prefix}_{k}" for k in MER3_NAMES]


def extract_kmer2_freq(seq_num: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(MER2_NAMES), dtype=np.float32)
    total = 0.0
    for i in range(len(seq_num) - 1):
        b1, b2 = int(seq_num[i]), int(seq_num[i + 1])
        if b1 < 0 or b2 < 0:
            continue
        key = NUCLEOTIDES_DNA[b1] + NUCLEOTIDES_DNA[b2]
        counts[MER2_MAP[key]] += 1.0
        total += 1.0
    if total > 0:
        counts /= total
    return counts


def extract_kmer3_freq(seq_num: np.ndarray, start: int = None, end: int = None) -> np.ndarray:
    """
    3-mer frequency in [start,end) over positions (start..end-3).
    If start/end not given, use whole sequence.
    """
    if start is None:
        start = 0
    if end is None:
        end = len(seq_num)
    start = max(0, int(start))
    end = min(len(seq_num), int(end))

    counts = np.zeros(len(MER3_NAMES), dtype=np.float32)
    total = 0.0
    for i in range(start, end - 2):
        b1, b2, b3 = int(seq_num[i]), int(seq_num[i + 1]), int(seq_num[i + 2])
        if b1 < 0 or b2 < 0 or b3 < 0:
            continue
        key = NUCLEOTIDES_DNA[b1] + NUCLEOTIDES_DNA[b2] + NUCLEOTIDES_DNA[b3]
        counts[MER3_MAP[key]] += 1.0
        total += 1.0
    if total > 0:
        counts /= total
    return counts


# ------------------ IO helpers ------------------
def resolve_long_csv(base_dir: str, mode: str, long_csv: str) -> str:
    if long_csv and long_csv.strip():
        return long_csv.strip()
    mode = mode.lower().strip()
    return os.path.join(base_dir, f"ALL_cells_{mode}_CpG_long_table.csv")


def default_out_csv(out_dir: str, mode: str) -> str:
    mode = mode.lower().strip()
    return os.path.join(out_dir, f"seq_features_from_{mode}.csv")


# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser(description="P2a: static sequence features from long table (dedup to site-level)")
    ap.add_argument("--base_dir", default=r"data/data_out/PBMC_CPG_ALL_UNION")
    ap.add_argument("--input_mode", choices=["union", "intersection"], default="union")
    ap.add_argument("--long_csv", default="", help="optional manual long table path (P1 output)")
    ap.add_argument("--genome_fasta", default="data/hg38.fa", help="reference genome FASTA path (e.g., mm10.fa or GRCm38.fa)")
    ap.add_argument("--out_dir", default=r"feature_output/PBMC_CPG_ALL_UNION")
    ap.add_argument("--out_csv", default="", help="optional output csv path")

    # sequence extraction
    ap.add_argument("--seq_window_radius", type=int, default=100,
                    help="flanking window radius around CpG center (default 100 => 201bp)")

    # feature toggles (kept consistent with your A2 script)
    ap.add_argument("--disable_bio", action="store_true", help="disable BIO_* block")
    ap.add_argument("--pos_radius", type=int, default=10, help="POS_* one-hot radius (default 10 => 21bp excluding center)")
    ap.add_argument("--disable_pos", action="store_true", help="disable POS_* block")
    ap.add_argument("--enable_kmer2", action="store_true", help="enable kmer2_* (16 dims)")
    ap.add_argument("--enable_center_kmer3", action="store_true", help="enable center-window kmer3 (64 dims)")
    ap.add_argument("--center_kmer3_radius", type=int, default=10, help="center window radius for kmer3c (default 10)")

    args = ap.parse_args()

    long_csv = resolve_long_csv(args.base_dir, args.input_mode, args.long_csv)
    if not os.path.exists(long_csv):
        raise FileNotFoundError(f"Long table not found: {long_csv}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = args.out_csv.strip() if args.out_csv.strip() else default_out_csv(args.out_dir, args.input_mode)

    if not os.path.exists(args.genome_fasta):
        raise FileNotFoundError(f"Genome FASTA not found: {args.genome_fasta}")

    logger.info("=" * 90)
    logger.info("🧬 P2a: Extract STATIC sequence features")
    logger.info("=" * 90)
    logger.info(f"Input long : {long_csv}")
    logger.info(f"Genome     : {args.genome_fasta}")
    logger.info(f"Out CSV    : {out_csv}")
    logger.info(f"Window     : radius={args.seq_window_radius} (len={2 * args.seq_window_radius + 1})")

    # 1) load long table
    df_long = pd.read_csv(long_csv, low_memory=False)
    required = {"chrom", "pos_based"}
    miss = required - set(df_long.columns)
    if miss:
        raise ValueError(f"Long table missing columns: {miss}")

    # 2) clean
    df_long["chrom"] = df_long["chrom"].astype(str).str.strip()
    df_long["pos_based"] = pd.to_numeric(df_long["pos_based"], errors="coerce").astype("Int64")
    df_long = df_long[df_long["pos_based"].notna() & (df_long["pos_based"] > 0)].reset_index(drop=True)

    # 3) dedup to site-level
    df_site = df_long[["chrom", "pos_based"]].drop_duplicates().reset_index(drop=True)
    logger.info(f"Long rows={len(df_long):,} | unique sites={len(df_site):,}")

    # 4) parse genome
    logger.info("Parsing genome FASTA ...")
    chr_seq_dict = parse_genome_fasta(args.genome_fasta)
    logger.info(f"Loaded {len(chr_seq_dict)} contigs/chromosomes.")

    # 5) extract flanking sequences
    logger.info("Extracting flanking sequences ...")
    seqs = []
    for _, row in tqdm(df_site.iterrows(), total=len(df_site), desc="Extract flanking seq"):
        chrom = row["chrom"]
        pos = int(row["pos_based"])
        seq = get_flanking_seq(chr_seq_dict, chrom, pos, args.seq_window_radius)
        seqs.append(seq)
    df_site["Sequence"] = seqs

    # 6) encode sequences
    seq_numeric_list = [seq_to_numeric(s) for s in df_site["Sequence"].tolist()]

    # 7) build feature names
    feat_names = []
    enabled_blocks = []

    if not args.disable_bio:
        enabled_blocks.append("bio")
        feat_names.extend(bio_feature_names())

    if not args.disable_pos:
        enabled_blocks.append("pos")
        feat_names.extend(pos_feature_names(args.pos_radius))

    enabled_blocks.append("kmer3_global")
    feat_names.extend(kmer3_feature_names("kmer3"))

    if args.enable_center_kmer3:
        enabled_blocks.append("kmer3_center")
        feat_names.extend(kmer3_feature_names("kmer3c"))

    if args.enable_kmer2:
        enabled_blocks.append("kmer2")
        feat_names.extend(kmer2_feature_names())

    logger.info(f"Enabled blocks: {enabled_blocks}")
    logger.info(f"Feature dims (without keys): {len(feat_names)}")

    # 8) extract features
    feat_arrays = []
    for seq_num in tqdm(seq_numeric_list, total=len(seq_numeric_list), desc="Extract seq features"):
        n = len(seq_num)
        center = n // 2

        row_parts = []

        if not args.disable_bio:
            row_parts.append(extract_bio_features(seq_num))

        if not args.disable_pos:
            row_parts.append(extract_positional_onehot(seq_num, radius=args.pos_radius))

        # global 3-mer
        row_parts.append(extract_kmer3_freq(seq_num))

        # center-window 3-mer
        if args.enable_center_kmer3:
            r = int(args.center_kmer3_radius)
            start = center - r
            end = center + r + 1
            row_parts.append(extract_kmer3_freq(seq_num, start=start, end=end))

        if args.enable_kmer2:
            row_parts.append(extract_kmer2_freq(seq_num))

        feat_arrays.append(np.concatenate(row_parts, axis=0).astype(np.float32))

    X = np.vstack(feat_arrays).astype(np.float32)
    feat_df = pd.DataFrame(X, columns=feat_names)

    out_df = df_site[["chrom", "pos_based"]].copy()
    out_df = pd.concat([out_df, feat_df], axis=1)

    # 9) save
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    logger.info(f"✅ Saved seq features: {out_csv}")
    logger.info(f"Shape: {out_df.shape} (rows={len(out_df):,}, cols={len(out_df.columns):,})")
    logger.info(f"Example cols: {feat_names[:8]}")

if __name__ == "__main__":
    main()