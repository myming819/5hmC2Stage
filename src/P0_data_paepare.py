#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A0_cpg_intersection_cdhit_pipeline.py（适配仅4列：chrom/pos_c_1based/5hmc_signal/5mc_signal）
核心功能：
1) 读取11个细胞类型的CSV（仅含chrom/pos_c_1based/5hmc_signal/5mc_signal）
2) 按 (chrom, pos_c_1based) 求跨细胞并集(union) + 交集(intersection)
3) 计算每个位点的5hmc覆盖度（非NaN的细胞数量）
4) 科学选择覆盖度k值（Elbow Point法）+ 筛选高置信度位点
5) 科学计算5hmc阈值（GMM高斯混合模型）+ 添加全局标签
6) 输出：基础列 + 覆盖度列 + 标签列 + 详细统计（正负比/信号比例）
   - 同时输出union和intersection两个版本的文件及统计
   - 每一步处理都输出位点数量，追踪数据变化
"""

import os
import argparse
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

# # brain的11个细胞类型（对应11组5hmc/5mc列）
# CELL_TYPES = ['OPC', 'ODC1', 'ODC2', 'ODC3', 'MGC', 'INH', 'ENDO', 'ASC1', 'ASC2', 'EXC1', 'EXC2']
# PBMC的5个细胞类型（对应11组5hmc/5mc列）
CELL_TYPES = ['B', 'T_reg', 'T_naive', 'NK', 'Monocytes']
# mESC的2个细胞类型（对应11组5hmc/5mc列）
# CELL_TYPES = ['serum.bs', '2i.bs']
# 最终基础输出列（严格适配你的4列结构）
BASE_COLUMNS = ['chrom', 'pos_based'] + \
               [f'5hmc_signal_{cell}' for cell in CELL_TYPES] + \
               [f'5mc_signal_{cell}' for cell in CELL_TYPES]


def load_cell_csv(path: str, cell: str):
    """加载单个细胞CSV（仅校验你指定的4列）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{cell} 文件不存在: {path}")

    # 读取原始数据，统计初始行数
    df_raw = pd.read_csv(path, low_memory=False)
    raw_rows = len(df_raw)
    print(f"\n📥 [{cell}] 原始数据行数：{raw_rows:,}")

    # ========== 仅校验你指定的4列 ==========
    required_cols = {"chrom", "pos_c_1based", "5hmc_signal", "5mc_signal"}
    missing_cols = required_cols - set(df_raw.columns)
    if missing_cols:
        raise ValueError(f"{cell} 缺少必要列: {missing_cols}（仅支持 chrom/pos_c_1based/5hmc_signal/5mc_signal）")

    # 数据清洗：删除chrom/pos缺失值（信号0/NaN保留） + 去重
    df = df_raw[["chrom", "pos_c_1based", "5hmc_signal", "5mc_signal"]].copy()
    before_clean_rows = len(df)
    df = df.dropna(subset=["chrom", "pos_c_1based"])
    after_na_drop_rows = len(df)
    df = df.drop_duplicates(subset=["chrom", "pos_c_1based"], keep="first")
    after_dedup_rows = len(df)

    # 打印清洗过程的位点数量变化
    print(f"🧹 [{cell}] 数据清洗位点变化：")
    print(f"   - 筛选核心列后：{before_clean_rows:,} 条")
    print(f"   - 删除chrom/pos缺失值后：{after_na_drop_rows:,} 条")
    print(f"   - 去重后最终有效位点：{after_dedup_rows:,} 条")

    # 类型转换（避免建模时数值错误）
    df["pos_c_1based"] = df["pos_c_1based"].astype(int)
    df["5hmc_signal"] = pd.to_numeric(df["5hmc_signal"], errors="raise")
    df["5mc_signal"] = pd.to_numeric(df["5mc_signal"], errors="raise")

    # 重命名列：为每个细胞添加专属后缀（如5hmc_signal_OPC）
    df = df.rename(columns={
        "pos_c_1based": "pos_based",  # 统一pos列名
        "5hmc_signal": f"5hmc_signal_{cell}",
        "5mc_signal": f"5mc_signal_{cell}"
    })

    return df


def calculate_coverage(df: pd.DataFrame):
    """
    计算每个位点的5hmc覆盖度（非NaN的细胞数量）
    :param df: 合并后的位点数据
    :return: 添加coverage_hmc列的df, 覆盖度分布Series
    """
    before_coverage_rows = len(df)
    print(f"\n📏 计算覆盖度前位点数量：{before_coverage_rows:,}")

    # 1. 提取5hmc的信号列
    hmc_cols = [f"5hmc_signal_{cell}" for cell in CELL_TYPES]

    # 2. 计算覆盖度：非NaN的细胞数量（不管信号是0还是>0，只要非NaN就算覆盖）
    df["coverage_hmc"] = df[hmc_cols].notna().sum(axis=1)  # 5hmc覆盖度（非NaN细胞数）
    after_coverage_rows = len(df)
    print(f"📏 计算覆盖度后位点数量：{after_coverage_rows:,}（无变化，仅添加列）")

    # 3. 输出覆盖度统计（仅保留5hmc）
    print(f"\n📊 位点覆盖度统计：")
    print(f"   - 5hmc覆盖度分布（非NaN细胞数）：")
    hmc_coverage_dist = df["coverage_hmc"].value_counts().sort_index()
    print(hmc_coverage_dist)

    return df, hmc_coverage_dist


def find_optimal_k(coverage_dist: pd.Series, signal_cols: list, df: pd.DataFrame, signal_type: str):
    """
    科学选择最佳覆盖度k值（Elbow Point法）
    :param coverage_dist: 覆盖度分布（value_counts）
    :param signal_cols: 信号列列表（如5hmc_signal_*）
    :param df: 原始数据
    :param signal_type: hmc（仅支持hmc）
    :return: 最佳k值、k值分析结果
    """
    # 1. 生成k值列表（1到最大覆盖度）
    k_list = sorted(coverage_dist.index.tolist())
    k_analysis = []

    # 2. 对每个k，计算：保留位点比例、信号富集度（有信号位点占比）
    total_sites = len(df)
    print(f"\n🎯 开始k值分析，总位点数量：{total_sites:,}")

    # 提取所有非NaN的信号值（用于计算富集度）
    all_signal_values = df[signal_cols].values.flatten()
    all_signal_values = all_signal_values[~np.isnan(all_signal_values)]
    global_signal_ratio = (all_signal_values > 0).mean()  # 全局有信号比例

    for k in k_list:
        # 筛选覆盖度≥k的位点
        k_df = df[df[f"coverage_{signal_type}"] >= k]
        # 保留位点比例
        retain_ratio = len(k_df) / total_sites
        # 该k下的信号值（非NaN）
        k_signal_values = k_df[signal_cols].values.flatten()
        k_signal_values = k_signal_values[~np.isnan(k_signal_values)]
        # 信号富集度（该k下有信号比例 / 全局有信号比例）
        if len(k_signal_values) == 0:
            enrichment = 0
            signal_ratio = 0
        else:
            signal_ratio = (k_signal_values > 0).mean()
            enrichment = signal_ratio / global_signal_ratio if global_signal_ratio > 0 else 0

        k_analysis.append({
            "k": k,
            "retain_sites": len(k_df),
            "retain_ratio": retain_ratio,
            "signal_ratio": signal_ratio,
            "enrichment": enrichment
        })

    # 转换为DataFrame
    k_analysis_df = pd.DataFrame(k_analysis)
    print(f"\n🎯 {signal_type.upper()}覆盖度k值分析（各k值保留位点数量）：")
    print(k_analysis_df[["k", "retain_sites", "retain_ratio"]].round(4))

    # 找Elbow Point（富集度提升放缓，且保留比例>10%）
    # 计算富集度的一阶差分（变化率）
    k_analysis_df["enrichment_diff"] = k_analysis_df["enrichment"].diff().fillna(0)
    # 筛选：富集度变化率<0.1（提升放缓）且保留比例>0.1的最小k
    eligible_k = k_analysis_df[
        (k_analysis_df["enrichment_diff"] < 0.1) &
        (k_analysis_df["retain_ratio"] > 0.1)
        ]
    if not eligible_k.empty:
        optimal_k = eligible_k["k"].min()
        optimal_k_retain = eligible_k[eligible_k["k"] == optimal_k]["retain_sites"].values[0]
        print(f"\n✅ {signal_type.upper()}最佳k值（Elbow Point）：{optimal_k}（保留位点{optimal_k_retain:,}条）")
    else:
        optimal_k = 5  # 兜底默认值
        print(
            f"\n✅ {signal_type.upper()}最佳k值（兜底默认）：{optimal_k}（保留位点{k_analysis_df[k_analysis_df['k'] == 5]['retain_sites'].values[0]:,}条）")

    print(f"   - 选择依据：富集度提升放缓，且保留位点比例>10%")

    return optimal_k, k_analysis_df


def gmm_find_optimal_threshold(signal_values: np.ndarray, signal_type: str = "hmc"):
    """
    用高斯混合模型（GMM）找最佳信号阈值（区分背景/信号峰）
    :param signal_values: 非NaN的信号值数组
    :param signal_type: hmc/mc（用于输出）
    :return: 最佳阈值
    """
    # 1. 数据预处理：仅保留>0的值（背景峰通常在0附近）
    pre_filter_count = len(signal_values)
    signal_values = signal_values[signal_values >= 0]
    post_filter_count = len(signal_values)

    print(f"\n📈 GMM阈值分析 - {signal_type.upper()}信号值统计：")
    print(f"   - 原始非NaN信号值数量：{pre_filter_count:,}")
    print(f"   - 过滤≥0后信号值数量：{post_filter_count:,}")

    if len(signal_values) < 100:
        print(f"⚠️ {signal_type.upper()}信号值样本量不足（<100），使用默认阈值0.1")
        return 0.1

    # 2. 重塑数据（适配sklearn）
    X = signal_values.reshape(-1, 1)

    # 3. 拟合2组分GMM（背景峰+信号峰）
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)

    # 4. 找到两个峰的均值，取中间值作为阈值
    means = sorted(gmm.means_.flatten())
    threshold = (means[0] + means[1]) / 2

    # 5. 输出GMM分析结果
    print(f"\n🎯 {signal_type.upper()}阈值（GMM拟合）：")
    print(f"   - 背景峰均值：{means[0]:.4f}")
    print(f"   - 信号峰均值：{means[1]:.4f}")
    print(f"   - 最佳分割阈值：{threshold:.4f}")

    return threshold


def add_5hmc_label(df: pd.DataFrame, threshold: float = 0.0, label_type: str = "global"):
    """
    添加5hmc正负样本标签（适配你的列结构）
    :param threshold: 5hmc_signal>threshold为正样本(1)，否则为负样本(0)
    :param label_type: 仅支持global（全局标签，符合你的需求）
    """
    before_label_rows = len(df)
    print(f"\n🏷️  添加5hmc标签前位点数量：{before_label_rows:,}")

    if label_type != "global":
        print("⚠️ 仅支持global标签类型，自动切换为global")
        label_type = "global"

    # 提取所有5hmc信号列
    hmc_cols = [f"5hmc_signal_{cell}" for cell in CELL_TYPES]
    # 全局标签：该位点在任意细胞中5hmc>阈值则为1（NaN不参与判断）
    df["5hmc_label_global"] = (df[hmc_cols] > threshold).any(axis=1).astype(int)

    after_label_rows = len(df)
    print(f"🏷️  添加5hmc标签后位点数量：{after_label_rows:,}（无变化，仅添加列）")

    # 计算正负样本统计
    pos_count = df["5hmc_label_global"].sum()
    neg_count = len(df) - pos_count
    pos_ratio = pos_count / len(df) if len(df) > 0 else 0

    print(f"\n🏷️  5hmc全局标签统计：")
    print(f"   - 正样本（> {threshold}）：{pos_count} 条 ({pos_ratio:.2%})")
    print(f"   - 负样本（≤ {threshold}）：{neg_count} 条 ({1 - pos_ratio:.2%})")

    return df, pos_count, neg_count, pos_ratio


def filter_high_confidence(df: pd.DataFrame, k_hmc: int):
    """
    筛选高置信度位点：仅基于5hmc覆盖度≥阈值的位点
    :param df: 带coverage_hmc列的位点数据
    :param k_hmc: 5hmc最小覆盖度（≥k个细胞有非NaN信号）
    :return: 高置信度位点数据
    """
    before_filter_rows = len(df)
    print(f"\n🔍 高置信度筛选前位点数量：{before_filter_rows:,}")
    print(f"🔍 筛选条件：5hmc覆盖度≥{k_hmc}")

    # 筛选条件：仅5hmc覆盖度≥k_hmc
    filter_cond = df["coverage_hmc"] >= k_hmc
    high_conf_df = df[filter_cond].copy().reset_index(drop=True)

    after_filter_rows = len(high_conf_df)
    filter_drop_count = before_filter_rows - after_filter_rows
    filter_drop_ratio = filter_drop_count / before_filter_rows if before_filter_rows > 0 else 0

    # 输出筛选统计
    print(f"\n🔍 高置信度位点筛选结果：")
    print(f"   - 筛选逻辑：5hmc覆盖度≥{k_hmc}")
    print(f"   - 原始位点总数：{before_filter_rows:,}")
    print(f"   - 高置信度位点数：{after_filter_rows:,}")
    print(f"   - 筛选掉的位点数：{filter_drop_count:,} ({filter_drop_ratio:.2%})")
    print(f"   - 保留比例：{after_filter_rows / before_filter_rows:.2%}")

    return high_conf_df


def post_process_and_save(df_in: pd.DataFrame,
                          name: str,
                          out_dir: str,
                          args,
                          hmc_cols: list,
                          coverage_dist: pd.Series):
    """
    对 union/intersection 共用：可选覆盖度筛选、可选GMM阈值打标签、保存csv、打印最终汇总
    """
    df = df_in.copy()
    initial_rows = len(df)
    print(f"\n" + "-" * 80)
    print(f"📋 [{name}] 开始后处理，初始位点数量：{initial_rows:,}")
    print("-" * 80)

    # 1) 可选：高置信度筛选（沿用现有逻辑）
    if args.filter_high_confidence:
        if args.k_hmc is None:
            optimal_k_hmc, _ = find_optimal_k(coverage_dist, hmc_cols, df, "hmc")
            k_hmc = optimal_k_hmc
        else:
            k_hmc = args.k_hmc
        print(f"\n📌 [{name}] 使用k_hmc={k_hmc}进行高置信度筛选")
        df = filter_high_confidence(df, k_hmc)

    # 2) 可选：GMM阈值 + global label
    pos_count, neg_count, pos_ratio = 0, 0, 0
    used_thr = None
    if args.add_5hmc_label:
        all_hmc_values = df[hmc_cols].values.flatten()
        all_hmc_values = all_hmc_values[~np.isnan(all_hmc_values)]
        if args.label_threshold is None:
            used_thr = gmm_find_optimal_threshold(all_hmc_values, "hmc")
        else:
            used_thr = args.label_threshold
        print(f"\n📌 [{name}] 使用5hmc阈值 thr={used_thr:.6f}添加标签")
        df, pos_count, neg_count, pos_ratio = add_5hmc_label(df, threshold=used_thr, label_type="global")

    # 3) 保存文件
    suffix_parts = []
    if args.filter_high_confidence:
        suffix_parts.append(f"high_conf_hmc{(args.k_hmc if args.k_hmc is not None else 'auto')}")
    if args.add_5hmc_label:
        suffix_parts.append(f"label_thresh{(used_thr if used_thr is not None else args.label_threshold):.4f}")
    suffix = "_".join(suffix_parts) if suffix_parts else "raw"

    out_path = os.path.join(out_dir, f"ALL_cells_{name}_CpG_4cols_{suffix}.csv")
    df.to_csv(out_path, index=False, na_rep="NaN")

    # 4) 终端汇总（强化位点数量统计）
    print(f"\n" + "=" * 80)
    print(f"📊 [{name.upper()}] 最终结果汇总")
    print("=" * 80)
    print(f"   - 初始处理位点数量：{initial_rows:,}")
    print(f"   - 最终输出位点数量：{len(df):,}")
    print(f"   - 输出列数：{len(df.columns)}")
    print(f"   - 输出文件：{out_path}")
    if "coverage_hmc" in df.columns:
        print(f"   - 最终5hmc平均覆盖度：{df['coverage_hmc'].mean():.2f}")
    if args.add_5hmc_label:
        print(f"   - 5hmc正样本数：{pos_count:,} ({pos_ratio:.2%})")
        print(f"   - 5hmc负样本数：{neg_count:,} ({1 - pos_ratio:.2%})")
        all_hmc_vals = df[hmc_cols].values.flatten()
        all_hmc_vals = all_hmc_vals[~np.isnan(all_hmc_vals)]
        has_signal_ratio = (all_hmc_vals > 0).mean() if len(all_hmc_vals) else 0
        print(f"   - 5hmc有信号位点比例（>0）：{has_signal_ratio:.2%}")
        print(f"   - 5hmc无信号位点比例（≤0）：{1 - has_signal_ratio:.2%}")
    print("=" * 80)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="跨细胞CpG合并（科学筛选高置信度位点+自动计算阈值）")
    parser.add_argument("--out-dir",
                        default="data/data_out/PBMC_CPG_ALL_UNION",
                        help="输出目录")
    parser.add_argument("--input-pattern",
                        default="data/data_out/PBMC_CPG_{cell}/core_intersection_model.csv",
                        help="输入文件路径模板（{cell}会替换为细胞名，需保证文件仅含指定4列）")
    # 5hmc标签参数（支持手动/自动阈值）
    parser.add_argument("--add-5hmc-label",
                        action="store_true",
                        help="是否添加5hmc正负样本标签")
    parser.add_argument("--label-threshold",
                        type=float,
                        default=None,
                        help="5hmc正样本阈值（手动指定，不指定则用GMM自动计算）")
    # 高置信度筛选参数（仅保留5hmc相关）
    parser.add_argument("--filter-high-confidence",
                        action="store_true",
                        help="是否筛选高置信度位点")
    parser.add_argument("--k-hmc",
                        type=int,
                        default=None,
                        help="5hmc最小覆盖度（手动指定，不指定则用Elbow Point自动计算）")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # ========== 1. 读取所有细胞的4列数据 ==========
    print("=" * 80)
    print("📥 开始读取各细胞数据")
    print("=" * 80)
    cell_dfs = {}
    total_cell_sites = 0
    for cell in CELL_TYPES:
        input_path = args.input_pattern.format(cell=cell)
        try:
            df = load_cell_csv(input_path, cell)
            cell_dfs[cell] = df
            cell_sites = len(df)
            total_cell_sites += cell_sites
            print(f"✅ {cell}: 读取成功，有效位点 {cell_sites:,} 条（累计：{total_cell_sites:,} 条）")
        except Exception as e:
            print(f"❌ {cell}: 读取失败 - {e}")
            exit(1)

    # ========== 2. 构建并集(union) 和 交集(intersection) ==========
    print("\n" + "=" * 80)
    print("🔗 开始构建跨细胞并集(Union)和交集(Intersection)")
    print("=" * 80)

    # union：outer join（保留所有位点）
    union_df = cell_dfs[CELL_TYPES[0]]
    print(f"\n🌐 [UNION] 初始细胞({CELL_TYPES[0]})位点数量：{len(union_df):,}")

    # intersection：inner join（只保留所有细胞共有的位点）
    inter_df = cell_dfs[CELL_TYPES[0]]
    print(f"🔹 [INTER] 初始细胞({CELL_TYPES[0]})位点数量：{len(inter_df):,}")

    for cell in CELL_TYPES[1:]:
        merge_cols = ["chrom", "pos_based", f"5hmc_signal_{cell}", f"5mc_signal_{cell}"]

        # 合并到union（outer join）
        union_before_merge = len(union_df)
        union_df = union_df.merge(
            cell_dfs[cell][merge_cols],
            on=["chrom", "pos_based"],
            how="outer"
        )
        union_after_merge = len(union_df)
        union_added = union_after_merge - union_before_merge
        print(f"\n🌐 [UNION] 合并{cell}：")
        print(f"   - 合并前：{union_before_merge:,} 条")
        print(f"   - 合并后：{union_after_merge:,} 条（新增 {union_added:,} 条）")

        # 合并到intersection（inner join）
        inter_before_merge = len(inter_df)
        inter_df = inter_df.merge(
            cell_dfs[cell][merge_cols],
            on=["chrom", "pos_based"],
            how="inner"
        )
        inter_after_merge = len(inter_df)
        inter_removed = inter_before_merge - inter_after_merge
        print(f"🔹 [INTER] 合并{cell}：")
        print(f"   - 合并前：{inter_before_merge:,} 条")
        print(f"   - 合并后：{inter_after_merge:,} 条（移除 {inter_removed:,} 条）")

    # ========== 3. 规范列顺序（保留原始NaN，不填充） ==========
    print("\n" + "=" * 80)
    print("🧮 规范列顺序（保证列顺序统一）")
    print("=" * 80)
    union_before_reindex = len(union_df)
    union_df = union_df.reindex(columns=BASE_COLUMNS).reset_index(drop=True)
    union_after_reindex = len(union_df)
    print(f"🌐 [UNION] 列规范前：{union_before_reindex:,} 条 | 规范后：{union_after_reindex:,} 条")

    inter_before_reindex = len(inter_df)
    inter_df = inter_df.reindex(columns=BASE_COLUMNS).reset_index(drop=True)
    inter_after_reindex = len(inter_df)
    print(f"🔹 [INTER] 列规范前：{inter_before_reindex:,} 条 | 规范后：{inter_after_reindex:,} 条")

    # ========== 4. 计算位点覆盖度（仅5hmc） ==========
    print("\n" + "=" * 80)
    print("📏 计算UNION位点覆盖度")
    print("=" * 80)
    union_df, hmc_coverage_dist_union = calculate_coverage(union_df)

    print("\n" + "=" * 80)
    print("📏 计算INTERSECTION位点覆盖度")
    print("=" * 80)
    inter_df, hmc_coverage_dist_inter = calculate_coverage(inter_df)

    # ========== 5. 提取5hmc信号列（后续处理共用） ==========
    hmc_cols = [f"5hmc_signal_{cell}" for cell in CELL_TYPES]

    # ========== 6. 对union和intersection分别进行后处理+保存+汇总 ==========
    # 处理并保存union结果
    print("\n" + "=" * 80)
    print("💾 开始处理并保存UNION结果")
    print("=" * 80)
    post_process_and_save(union_df, "union", args.out_dir, args, hmc_cols, hmc_coverage_dist_union)

    # 处理并保存intersection结果
    print("\n" + "=" * 80)
    print("💾 开始处理并保存INTERSECTION结果")
    print("=" * 80)
    post_process_and_save(inter_df, "intersection", args.out_dir, args, hmc_cols, hmc_coverage_dist_inter)

    print("\n🎉 所有处理完成！")


if __name__ == "__main__":
    main()