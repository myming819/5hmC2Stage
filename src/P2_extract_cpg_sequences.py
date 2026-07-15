#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 P4 Stage1 表中读取真正用于一阶段训练的正负位点，
提取上下游序列并生成用于 MEME 的正负 FASTA。

特点：
1. 输入改为 P4 Stage1 表（至少含 chrom, pos_based, label_site）
2. 自动把 label_site 映射成 site_role（1=pos, 0=neg）
3. 默认提取上下各100bp序列（总长201bp）
4. 默认去重
5. 默认去掉含N的序列
6. 默认正负样本分别随机抽样1000条
7. 若表中存在 gene_id/gene_name/dist_to_TSS，则保留到输出 csv 里
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm


# ------------------ FASTA解析 ------------------
def parse_genome_fasta(fasta_path):
    chr_seq = {}
    cur = None
    parts = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    chr_seq[cur] = "".join(parts).upper()
                cur = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if cur is not None:
            chr_seq[cur] = "".join(parts).upper()
    return chr_seq


# ------------------ 提取侧翼序列 ------------------
def get_flanking_seq(chr_seq_dict, chrom, pos_1based, window_radius):
    r = int(window_radius)
    total_len = 2 * r + 1

    if chrom not in chr_seq_dict:
        return "N" * total_len

    seq = chr_seq_dict[chrom]
    n = len(seq)
    center0 = int(pos_1based) - 1
    left0 = center0 - r
    right0 = center0 + r

    out = []
    for i in range(left0, right0 + 1):
        out.append(seq[i] if 0 <= i < n else "N")
    return "".join(out)


# ------------------ 写入FASTA ------------------
def write_fasta(sub_df, out_fa):
    out_dir = os.path.dirname(out_fa)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_fa, "w", encoding="utf-8") as f:
        for i, row in enumerate(sub_df.itertuples(index=False), start=1):
            header = f">{row.chrom}:{int(row.pos_based)}|{row.site_role}|id{i}"
            f.write(f"{header}\n{row.Sequence.upper()}\n")


# ------------------ 按类别随机抽样 ------------------
def sample_by_class(df, sample_n_per_class=1000, seed=42):
    parts = []
    for label in ["pos", "neg"]:
        sub = df[df["site_role"] == label].copy()
        if len(sub) > sample_n_per_class:
            sub = sub.sample(n=sample_n_per_class, random_state=seed)
        parts.append(sub)
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


# ------------------ label_site -> site_role ------------------
def build_site_role_from_label(df):
    x = pd.to_numeric(df["label_site"], errors="coerce")
    df = df.loc[x.notna()].copy()
    x = x.loc[x.notna()].astype(int)

    role = x.map({1: "pos", 0: "neg"})
    df = df.loc[role.notna()].copy()
    df["site_role"] = role.loc[role.notna()].values
    df["label_site"] = pd.to_numeric(df["label_site"], errors="coerce").astype(int)
    return df


# ------------------ 主函数 ------------------
def main():
    ap = argparse.ArgumentParser(
        description="从P4 Stage1表提取±100bp序列并生成正负FASTA（默认各抽1000条）"
    )
    ap.add_argument("--stage1_csv", required=True,
                    help="P4 Stage1 表，如 stage1_site_static_union.csv")
    ap.add_argument("--genome_fasta", required=True, help="参考基因组FASTA")
    ap.add_argument("--window_radius", type=int, default=50,
                    help="序列窗口半径，默认100bp（总长201bp）")
    ap.add_argument("--sample_n_per_class", type=int, default=1000,
                    help="每类随机抽样数量，默认1000")
    ap.add_argument("--seed", type=int, default=123, help="随机种子")
    ap.add_argument("--pos_fa", required=True, help="正样本FASTA输出路径")
    ap.add_argument("--neg_fa", required=True, help="负样本FASTA输出路径")
    ap.add_argument("--out_seq_csv", default="stage1_sequence_for_meme.csv",
                    help="保存序列表格路径")
    ap.add_argument("--keep_n_sequences", action="store_true",
                    help="保留含N序列；默认去掉含N序列")
    args = ap.parse_args()

    # 1. 读取 P4 Stage1 表
    print("读取 P4 Stage1 表...")
    df = pd.read_csv(args.stage1_csv, low_memory=False)

    required_cols = {"chrom", "pos_based", "label_site"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"stage1_csv 缺少必要列: {missing}")

    df["chrom"] = df["chrom"].astype(str).str.strip()
    df["pos_based"] = pd.to_numeric(df["pos_based"], errors="coerce")
    df = df.dropna(subset=["chrom", "pos_based"])
    df["pos_based"] = df["pos_based"].astype(int)

    # 2. label_site -> site_role
    df = build_site_role_from_label(df)

    # 3. 去重（以一阶段真实训练位点为准）
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["chrom", "pos_based", "site_role"]).copy()
    after_dedup = len(df)
    print(f"去重完成：去掉 {before_dedup - after_dedup} 条重复记录，剩余 {after_dedup} 条")

    # 4. 加载参考基因组
    print("加载参考基因组...")
    chr_seq = parse_genome_fasta(args.genome_fasta)
    print(f"参考基因组载入完成：{len(chr_seq)} 条染色体/contig")

    # 5. 提取序列
    print("提取序列中...")
    seqs = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        seq = get_flanking_seq(chr_seq, row.chrom, row.pos_based, args.window_radius)
        seqs.append(seq)
    df["Sequence"] = seqs

    # 6. 去掉含N的序列（默认）
    if not args.keep_n_sequences:
        before_rm_n = len(df)
        df = df[~df["Sequence"].str.contains("N", na=False)].copy()
        after_rm_n = len(df)
        print(f"去除含N序列：去掉 {before_rm_n - after_rm_n} 条，剩余 {after_rm_n} 条")

    # 7. 正负分别抽样
    if args.sample_n_per_class > 0:
        before_sample = len(df)
        df = sample_by_class(df, sample_n_per_class=args.sample_n_per_class, seed=args.seed)
        after_sample = len(df)
        print(f"按类别抽样完成：从 {before_sample} 条变为 {after_sample} 条")
    else:
        print("未进行抽样，保留全部序列")

    # 8. 打乱顺序
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 9. 输出 csv
    out_dir = os.path.dirname(args.out_seq_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    base_cols = ["chrom", "pos_based", "label_site", "site_role"]
    extra_cols = [c for c in ["gene_id", "gene_name", "dist_to_TSS"] if c in df.columns]
    out_cols = base_cols + extra_cols + ["Sequence"]

    out_df = df[out_cols].copy()
    out_df.to_csv(args.out_seq_csv, index=False)
    print(f"序列表格已保存：{args.out_seq_csv}")

    # 10. 输出 FASTA
    pos_df = df[df["site_role"] == "pos"].copy()
    neg_df = df[df["site_role"] == "neg"].copy()

    write_fasta(pos_df, args.pos_fa)
    write_fasta(neg_df, args.neg_fa)

    # 11. 统计
    n_pos = len(pos_df)
    n_neg = len(neg_df)
    seq_len = 2 * args.window_radius + 1

    print("\n完成！")
    print(f"窗口长度：{seq_len} bp")
    print(f"正样本：{n_pos}")
    print(f"负样本：{n_neg}")
    print(f"正样本FASTA：{args.pos_fa}")
    print(f"负样本FASTA：{args.neg_fa}")


if __name__ == "__main__":
    main()