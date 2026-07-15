#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
P2c_static_tf_features.py
Stage2c: Build gene-level TF motif features from promoter FASTA + JASPAR PFM.

Input:
  - long table with gene_id (from P2b):
      data/data_out/Brain_CPG_ALL_UNION/ALL_cells_{mode}_CpG_long_table_with_gene.csv

Output:
  - feature_output/Brain_CPG_ALL_UNION/gene_TFlist_maxscore_topK{K}.csv
"""

import os
import re
import logging
import argparse
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from numba import jit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def norm_ensembl_gid(g: str) -> str:
    g = str(g).strip()
    if g in {"", "nan", "None"}:
        return ""
    return g.split(".")[0]

def detect_species_from_gene_ids(gene_ids_raw):
    tags = set()
    for g in gene_ids_raw:
        gb = norm_ensembl_gid(g)
        if gb.startswith("ENSG"):
            tags.add("human")
        elif gb.startswith("ENSMUSG"):
            tags.add("mouse")
    return tags

@jit(nopython=True)
def encode_seq_to_idx_numba(seq_arr):
    out = np.empty(len(seq_arr), dtype=np.int8)
    for i in range(len(seq_arr)):
        b = seq_arr[i]
        if b == 65:
            out[i] = 0
        elif b == 67:
            out[i] = 1
        elif b == 71:
            out[i] = 2
        elif b == 84:
            out[i] = 3
        else:
            out[i] = -1
    return out

def encode_seq_to_idx(seq: str) -> np.ndarray:
    seq = (seq or "").upper()
    seq_arr = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    return encode_seq_to_idx_numba(seq_arr)

def parse_jaspar_pfms(pfm_path: str):
    motifs = []
    cur_id, cur_name = None, None
    mat = []
    with open(pfm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None and len(mat) == 4:
                    motifs.append({"id": cur_id, "name": cur_name, "pfm": np.array(mat, dtype=np.float32)})
                header = line[1:].strip()
                parts = header.split()
                cur_id = parts[0]
                cur_name = parts[1] if len(parts) > 1 else cur_id
                mat = []
            else:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                row = [float(x) for x in nums]
                mat.append(row)
        if cur_id is not None and len(mat) == 4:
            motifs.append({"id": cur_id, "name": cur_name, "pfm": np.array(mat, dtype=np.float32)})
    return motifs

def pwm_to_ppm(pfm: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    pfm = pfm.astype(np.float32)
    ppm = pfm + float(pseudocount)
    colsum = ppm.sum(axis=0, keepdims=True)
    return ppm / colsum

def filter_motifs(motifs, min_ic=8.0, max_entropy=6.0, min_c_freq=0.05):
    """
    Keep motifs with sufficient information content / limited entropy, and CpG-relevant.
    """
    kept = []
    for m in motifs:
        ppm = pwm_to_ppm(m["pfm"], pseudocount=1.0)
        # IC per column
        eps = 1e-8
        ent = -np.sum(ppm * np.log(ppm + eps), axis=0)
        ic = np.log(4.0) - ent
        if float(ic.sum()) < float(min_ic):
            continue
        if float(ent.sum()) > float(max_entropy):
            continue
        # require some C frequency
        if float(ppm[1].mean()) < float(min_c_freq):
            continue
        kept.append(m)
    # sort by IC descending
    def ic_total(m):
        ppm = pwm_to_ppm(m["pfm"], pseudocount=1.0)
        eps = 1e-8
        ent = -np.sum(ppm * np.log(ppm + eps), axis=0)
        ic = np.log(4.0) - ent
        return float(ic.sum())
    kept.sort(key=ic_total, reverse=True)
    return kept

def calculate_background_freq(gene2seqidx: dict):
    cnt = np.zeros(4, dtype=np.float64)
    total = 0
    for _, seq_idx in gene2seqidx.items():
        for b in seq_idx:
            if b >= 0:
                cnt[b] += 1
                total += 1
    if total == 0:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    bg = (cnt / total).astype(np.float32)
    bg = bg / bg.sum()
    return bg

@jit(nopython=True)
def scan_max_score_numba(seq_idx: np.ndarray, log_pwm: np.ndarray):
    L = len(seq_idx)
    W = log_pwm.shape[1]
    best = -1e30
    for i in range(L - W + 1):
        s = 0.0
        ok = True
        for j in range(W):
            b = seq_idx[i + j]
            if b < 0:
                ok = False
                break
            s += log_pwm[b, j]
        if ok and s > best:
            best = s
    return best

def scan_max_score(seq_idx: np.ndarray, ppm: np.ndarray, bg: np.ndarray):
    eps = 1e-8
    pwm = ppm / (bg.reshape(4, 1) + eps)
    log_pwm = np.log(pwm + eps).astype(np.float32)
    return float(scan_max_score_numba(seq_idx.astype(np.int8), log_pwm))

def build_gene2seqidx_from_fasta(fasta_path: str, gene_base_set: set):
    """
    promoter FASTA header: >{gene_id}|{gene_name}|chr:start-end(strand)
    use header.split('|')[0] as gene_id, then remove version to match gene_base_set.
    """
    fa = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    gene2seqidx = {}
    for header in fa.keys():
        gid_raw = header.split("|")[0].strip()
        gid_base = norm_ensembl_gid(gid_raw)
        if gid_base in gene_base_set:
            seq = str(fa[header])
            seq_idx = encode_seq_to_idx(seq)
            if (seq_idx >= 0).any():
                gene2seqidx[gid_base] = seq_idx
    return gene2seqidx

def resolve_train_long(base_dir: str, mode: str, train_long: str) -> str:
    if train_long and train_long.strip():
        return train_long.strip()
    mode = mode.lower().strip()
    return os.path.join(base_dir, f"ALL_cells_{mode}_CpG_long_table_with_gene.csv")

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=r"data/data_out/PBMC_CPG_ALL_UNION")
    ap.add_argument("--input_mode", choices=["union", "intersection"], default="union")
    ap.add_argument("--train_long", default="", help="P2b 输出 long_with_gene；为空则 base_dir+mode 自动拼")
    ap.add_argument("--promoter_fasta_human", default="data/hg38_promoters_500bp.fa")
    ap.add_argument("--promoter_fasta_mouse", default="data/mm10_promoters_500bp.fa")
    ap.add_argument("--jaspar_pfm", default="data/JASPAR2022_CORE_vertebrates_non-redundant_pfms_jaspar.txt")
    ap.add_argument("--out_dir", default="feature_output/PBMC_CPG_ALL_UNION")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--min_ic", type=float, default=8.0)
    ap.add_argument("--max_entropy", type=float, default=6.0)
    ap.add_argument("--min_c_freq", type=float, default=0.05)
    ap.add_argument("--force_species", choices=["auto", "human", "mouse"], default="human")
    return ap.parse_args()

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    train_long = resolve_train_long(args.base_dir, args.input_mode, args.train_long)
    if not os.path.exists(train_long):
        raise FileNotFoundError(train_long)

    logger.info(f"读取 long_with_gene：{train_long}")
    df = pd.read_csv(train_long, usecols=["gene_id"], low_memory=False)
    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    gene_ids_raw = df["gene_id"].dropna().unique().tolist()
    if len(gene_ids_raw) == 0:
        raise ValueError("gene_id 为空：请先运行 P2b_map_site_to_gene.py")

    base2raw = {}
    for g in gene_ids_raw:
        gb = norm_ensembl_gid(g)
        if gb and gb not in base2raw:
            base2raw[gb] = g
    gene_base_set = set(base2raw.keys())
    logger.info(f"有效 gene_base 数量：{len(gene_base_set)}")

    # detect species
    if args.force_species == "auto":
        tags = detect_species_from_gene_ids(gene_ids_raw)
    else:
        tags = {args.force_species}
    if not tags:
        raise ValueError("无法识别物种：gene_id 既非 ENSG 也非 ENSMUSG。请检查 gene_id。")

    # load promoter FASTA (can be human/mouse/mixed)
    gene2seqidx = {}
    if "human" in tags:
        if not os.path.exists(args.promoter_fasta_human):
            raise FileNotFoundError(f"human promoter fasta not found: {args.promoter_fasta_human}")
        logger.info(f"加载 human promoter FASTA: {args.promoter_fasta_human}")
        gene2seqidx.update(build_gene2seqidx_from_fasta(args.promoter_fasta_human, gene_base_set))
    if "mouse" in tags:
        if not os.path.exists(args.promoter_fasta_mouse):
            raise FileNotFoundError(f"mouse promoter fasta not found: {args.promoter_fasta_mouse}")
        logger.info(f"加载 mouse promoter FASTA: {args.promoter_fasta_mouse}")
        gene2seqidx.update(build_gene2seqidx_from_fasta(args.promoter_fasta_mouse, gene_base_set))

    logger.info(f"promoter 匹配并编码成功：{len(gene2seqidx)}")
    if len(gene2seqidx) == 0:
        raise ValueError("无 gene_id 匹配到 promoter FASTA。请检查 promoter FASTA / gene_id 版本。")

    if not os.path.exists(args.jaspar_pfm):
        raise FileNotFoundError(args.jaspar_pfm)
    logger.info(f"解析 JASPAR PFM：{args.jaspar_pfm}")
    motifs = parse_jaspar_pfms(args.jaspar_pfm)
    if not motifs:
        raise ValueError("未解析到任何 motif")

    filtered = filter_motifs(motifs, min_ic=args.min_ic, max_entropy=args.max_entropy, min_c_freq=args.min_c_freq)
    if not filtered:
        raise ValueError("motif 筛选后为 0，请放宽 min_ic/max_entropy/min_c_freq")

    topk = min(args.top_k, len(filtered))
    selected = filtered[:topk]
    logger.info(f"最终 motif 数量：{topk}")

    bg = calculate_background_freq(gene2seqidx)

    gene_base_sorted = sorted(gene2seqidx.keys())
    gene_id_out = [base2raw.get(gb, gb) for gb in gene_base_sorted]

    out = {"gene_id": gene_id_out}
    logger.info(f"开始计算 {len(gene_base_sorted)} genes × {len(selected)} motifs")

    for i, m in enumerate(selected, 1):
        ppm = pwm_to_ppm(m["pfm"], pseudocount=1.0)
        col = f"{m['id']}::{m['name']}"
        scores = []
        for gb in gene_base_sorted:
            s = scan_max_score(gene2seqidx[gb], ppm, bg)
            scores.append(float(s))
        out[col] = scores
        if i % 10 == 0 or i == len(selected):
            logger.info(f"motif 扫描进度：{i}/{len(selected)}")

    df_out = pd.DataFrame(out)
    out_csv = os.path.join(args.out_dir, f"gene_TFlist_maxscore_topK{topk}.csv")
    df_out.to_csv(out_csv, index=False)
    logger.info(f"输出完成：{out_csv} | shape={df_out.shape}")

if __name__ == "__main__":
    args = get_args()
    main(args)