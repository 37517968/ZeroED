import argparse
import ast
from contextlib import nullcontext
import json
import multiprocessing
import os
import pickle
import random
import re
import time
import shutil
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from scipy import stats

import fasttext.util
import numpy as np
import pandas as pd
# import torch.multiprocessing as mp
import yaml
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from feature import cluster, feat_gen_df, feat_gen_df_incremental, feat_gen_global_cache
from get_rel_attrs import (cal_all_column_nmi, cal_strong_res_column_nmi)
from measure import measure_detect
from prompt_gen import (create_err_gen_inst_prompt, err_clean_func_prompt,
                        error_check_prompt, pre_func_prompt,
                        create_clean_gen_inst_prompt, create_dirty_gen_inst_prompt,
                        guide_gen_prompt, distribution_analysis_decision_prompt,
                        canonical_pattern_analysis_prompt, error_check_with_canonical_prompt,
                        llm_canonicality_score_prompt, llm_compare_patterns_canonicality_prompt
                        )
from utility import (Logger, Timer, copy_file,
                     default_dict_of_lists, get_ans_from_llm, query_base,
                     rag_query, split_list_to_sublists)


# ==================== 新增辅助函数 ====================

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# ==================== 分布分析方法相关函数 ====================

def get_single_column_features(dirty_csv, col_num, col_name):
    """
    获取单列特征（不考虑其他相关列）
    重点关注"形式/模式"相似性而非内容相似性
    这样像 movies_actor 这种列，形式相似的演员列表会聚在一起
    
    Args:
        dirty_csv: 脏数据DataFrame
        col_num: 列索引
        col_name: 列名
    
    Returns:
        features: 特征数组
    """
    from feature import str_agg, L2_str_agg, L3_str_agg
    from sklearn.preprocessing import MinMaxScaler
    from collections import Counter
    
    dirty_csv = dirty_csv.astype(str).fillna('nan')
    
    feature_list = []
    
    for row in range(len(dirty_csv)):
        feature = []
        val = str(dirty_csv.iloc[row, col_num])
        
        # ========== 形式/模式特征（主要特征）==========
        
        # 1. 基本结构特征
        feature.append(len(val))  # 字符串长度
        feature.append(len(val.split()))  # 单词数量
        feature.append(val.count(','))  # 逗号数量（列表分隔符）
        feature.append(val.count(';'))  # 分号数量
        feature.append(val.count('|'))  # 竖线数量
        feature.append(val.count('/'))  # 斜杠数量
        feature.append(val.count('-'))  # 连字符数量
        feature.append(val.count('(') + val.count(')'))  # 括号数量
        feature.append(val.count('[') + val.count(']'))  # 方括号数量
        
        # 2. 字符类型比例特征
        total_len = max(len(val), 1)
        digit_count = sum(c.isdigit() for c in val)
        alpha_count = sum(c.isalpha() for c in val)
        upper_count = sum(c.isupper() for c in val)
        lower_count = sum(c.islower() for c in val)
        space_count = sum(c.isspace() for c in val)
        special_count = sum(not c.isalnum() and not c.isspace() for c in val)
        
        feature.append(digit_count / total_len)  # 数字比例
        feature.append(alpha_count / total_len)  # 字母比例
        feature.append(upper_count / total_len)  # 大写字母比例
        feature.append(lower_count / total_len)  # 小写字母比例
        feature.append(space_count / total_len)  # 空格比例
        feature.append(special_count / total_len)  # 特殊字符比例
        
        # 3. Pattern特征（形式抽象）
        pat_list = str_agg(val)
        # L2 pattern: 字符类型序列 (D=数字, L=字母, S=符号)
        l2_pat = L2_str_agg(val)
        # L3 pattern: 更细粒度的字符类型序列
        l3_pat = L3_str_agg(val)
        
        # Pattern长度特征
        for pat in pat_list:
            feature.append(len(pat))
        
        # Pattern中各类型段的数量
        feature.append(l2_pat.count('\\D'))  # 数字段数量
        feature.append(l2_pat.count('\\L'))  # 字母段数量
        feature.append(l2_pat.count('\\S'))  # 符号段数量
        
        # 4. 结构复杂度特征
        # 字符类型转换次数（衡量结构复杂度）
        transitions = 0
        prev_type = None
        for c in val:
            if c.isdigit():
                curr_type = 'D'
            elif c.isalpha():
                curr_type = 'L'
            else:
                curr_type = 'S'
            if prev_type is not None and curr_type != prev_type:
                transitions += 1
            prev_type = curr_type
        feature.append(transitions)
        feature.append(transitions / total_len if total_len > 0 else 0)  # 转换密度
        
        # 5. 特殊值标识特征
        val_lower = val.lower().strip()
        feature.append(1.0 if val_lower in ['', 'nan', 'null', 'none', 'n/a', 'na', '-', '--', 'unknown', 'missing'] else 0.0)
        feature.append(1.0 if len(val.strip()) == 0 else 0.0)  # 是否为空或只有空格
        
        # 6. 首尾字符类型特征
        if len(val) > 0:
            first_char = val[0]
            last_char = val[-1]
            feature.append(1.0 if first_char.isupper() else 0.0)
            feature.append(1.0 if first_char.isdigit() else 0.0)
            feature.append(1.0 if last_char.isdigit() else 0.0)
            feature.append(1.0 if last_char in '.!?;:' else 0.0)
        else:
            feature.extend([0.0, 0.0, 0.0, 0.0])
        
        feature_list.append(feature)
    
    # 归一化
    scaler = MinMaxScaler()
    feature_array = np.array(feature_list)
    feature_array = np.nan_to_num(feature_array)
    if len(feature_array) > 0:
        feature_array = scaler.fit_transform(feature_array)
    
    return feature_array


def single_column_dbscan_clustering(dirty_csv, col_num, col_name, eps=0.3, min_samples=2):
    """
    对单列进行DBSCAN聚类（不考虑其他相关列）
    
    Args:
        dirty_csv: 脏数据DataFrame
        col_num: 列索引
        col_name: 列名
        eps: DBSCAN的距离阈值ε
        min_samples: DBSCAN的最小样本数
    
    Returns:
        cluster_result: 聚类结果字典
    """
    from sklearn.cluster import DBSCAN
    
    # 获取单列特征
    features = get_single_column_features(dirty_csv, col_num, col_name)
    
    if len(features) == 0:
        return None
    
    # DBSCAN聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = dbscan.fit_predict(features)
    
    # 获取聚类数量（不包括噪声点-1）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # 为每个聚类找到中心点
    cluster_centers = []
    cluster_indices = []
    cluster_values = []
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_point_indices = np.where(cluster_mask)[0]
        cluster_points = features[cluster_mask]
        
        if len(cluster_points) == 0:
            continue
        
        cluster_mean = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - cluster_mean, axis=1)
        closest_idx_in_cluster = np.argmin(distances)
        center_idx = cluster_point_indices[closest_idx_in_cluster]
        
        cluster_centers.append(center_idx)
        cluster_indices.append(cluster_point_indices.tolist())
        values = [str(dirty_csv.iloc[idx, col_num]) for idx in cluster_point_indices]
        cluster_values.append(values)
    
    noise_indices = np.where(labels == -1)[0].tolist()
    
    cluster_result = {
        'n_clusters': n_clusters,
        'labels': labels.tolist(),
        'cluster_centers': cluster_centers,
        'cluster_indices': cluster_indices,
        'cluster_values': cluster_values,
        'noise_indices': noise_indices,
        'features': features
    }
    
    return cluster_result


def calculate_pattern_entropy(values):
    """计算一组值的pattern熵"""
    from feature import L2_str_agg
    import math
    
    if len(values) == 0:
        return 0.0
    
    patterns = [L2_str_agg(str(v)) for v in values]
    pattern_counts = Counter(patterns)
    total = len(patterns)
    
    entropy = 0.0
    for count in pattern_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p + 1e-10)
    
    return entropy


def calculate_string_similarity(s1, s2):
    """计算两个字符串的相似度（使用编辑距离）"""
    s1, s2 = str(s1), str(s2)
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    edit_distance = dp[m][n]
    max_len = max(m, n)
    
    if max_len == 0:
        return 1.0
    
    return 1 - edit_distance / max_len


def get_llm_canonicality_score(attr_name, sample_values, logger=None):
    """
    使用LLM评估聚类样本值的规范性
    
    Args:
        attr_name: 属性名称
        sample_values: 样本值列表
        logger: 日志记录器
    
    Returns:
        score: 规范性分数 (0-1)
    """
    # 最多取5个样本
    samples = sample_values[:5] if len(sample_values) > 5 else sample_values
    
    # 先进行快速的规则检查，对于明显的无效值直接返回低分
    invalid_patterns = ['', 'nan', 'null', 'none', 'n/a', 'na', '-', '--', 'unknown', 'missing', ' ']
    
    # 检查是否所有样本都是无效值
    invalid_count = 0
    for val in samples:
        val_lower = str(val).lower().strip()
        if val_lower in invalid_patterns or len(val_lower) == 0:
            invalid_count += 1
    
    # 如果大部分是无效值，直接返回低分
    if invalid_count >= len(samples) * 0.8:
        if logger:
            logger.info(f"列 '{attr_name}' 聚类样本大部分为无效值，LLM规范性分数: 0.1")
        return 0.1
    
    # 调用LLM评估
    prompt = llm_canonicality_score_prompt(attr_name, samples)
    
    try:
        response = query_base(prompt)
        response = response.strip()
        
        # 尝试解析分数
        try:
            score = float(response)
            score = max(0.0, min(1.0, score))  # 确保在0-1范围内
        except ValueError:
            # 如果无法解析，尝试从响应中提取数字
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                score = max(0.0, min(1.0, score))
            else:
                score = 0.5  # 默认中等分数
        
        if logger:
            logger.info(f"列 '{attr_name}' 聚类LLM规范性分数: {score:.2f}")
        
        return score
    except Exception as e:
        if logger:
            logger.warning(f"获取LLM规范性分数时出错: {str(e)}，使用默认分数0.5")
        return 0.5


def calculate_canonical_score(cluster_values, total_samples, alpha=0.25, beta=0.15, gamma=0.15, delta=0.45,
                              attr_name=None, logger=None, use_llm_score=True):
    """
    计算聚类的规范得分（Canonical Score）
    
    S_canon(C_j) = α * Freq(C_j) + β * Reg(C_j) + γ * Compact(C_j) + δ * LLM_Canon(C_j)
    
    参数说明：
    - α (alpha): 频率项权重，默认0.25
    - β (beta): 规则性项权重，默认0.15
    - γ (gamma): 紧致性项权重，默认0.15
    - δ (delta): LLM规范性项权重，默认0.45（较大权重，确保空值等明显错误不会获得高分）
    
    Args:
        cluster_values: 聚类中的所有值列表
        total_samples: 总样本数N
        alpha, beta, gamma, delta: 各项权重
        attr_name: 属性名称（用于LLM评估）
        logger: 日志记录器
        use_llm_score: 是否使用LLM规范性评分
    
    Returns:
        score: 规范得分
        components: 各分量的值
    """
    import math
    
    cluster_size = len(cluster_values)
    
    if cluster_size == 0:
        return 0.0, {'freq': 0, 'reg': 0, 'compact': 0, 'llm_canon': 0}
    
    # (1) 频率项 Freq(C_j) = |C_j| / N
    freq = cluster_size / total_samples
    
    # (2) 规则性项 Reg(C_j) = 1 - pattern_entropy(C_j) / log(K)
    pattern_entropy = calculate_pattern_entropy(cluster_values)
    K = max(cluster_size, 2)
    log_K = math.log(K + 1e-10)
    reg = 1 - pattern_entropy / log_K if log_K > 0 else 1.0
    reg = max(0, min(1, reg))
    
    # (3) 紧致性项 Compact(C_j)
    if cluster_size <= 1:
        compact = 1.0
    else:
        # 为了效率，只采样部分值计算相似度
        sample_size = min(cluster_size, 20)
        if cluster_size > sample_size:
            import random
            sampled_values = random.sample(cluster_values, sample_size)
        else:
            sampled_values = cluster_values
        
        total_sim = 0.0
        for i in range(len(sampled_values)):
            for j in range(len(sampled_values)):
                total_sim += calculate_string_similarity(sampled_values[i], sampled_values[j])
        compact = total_sim / (len(sampled_values) * len(sampled_values))
    
    # (4) LLM规范性项 LLM_Canon(C_j)
    llm_canon = 0.5  # 默认中等分数
    if use_llm_score and attr_name:
        # 取样本值进行LLM评估
        sample_for_llm = cluster_values[:5] if len(cluster_values) > 5 else cluster_values
        llm_canon = get_llm_canonicality_score(attr_name, sample_for_llm, logger)
    
    # 计算总分
    # 如果不使用LLM分数，重新分配权重
    if use_llm_score:
        score = alpha * freq + beta * reg + gamma * compact + delta * llm_canon
    else:
        # 不使用LLM时，将delta的权重分配给其他项
        adjusted_alpha = alpha + delta / 3
        adjusted_beta = beta + delta / 3
        adjusted_gamma = gamma + delta / 3
        score = adjusted_alpha * freq + adjusted_beta * reg + adjusted_gamma * compact
        llm_canon = 0.0
    
    components = {
        'freq': freq,
        'reg': reg,
        'compact': compact,
        'llm_canon': llm_canon,
        'pattern_entropy': pattern_entropy
    }
    
    return score, components


def calculate_canonical_probability(scores):
    """计算每个聚类成为Canonical簇的概率（softmax）"""
    if len(scores) == 0:
        return []
    
    exp_scores = np.exp(np.array(scores) - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)
    
    return probabilities.tolist()


def get_cluster_pattern_description(cluster_values):
    """
    获取聚类的模式描述
    
    Args:
        cluster_values: 聚类中的值列表
    
    Returns:
        pattern_description: 模式描述字符串
    """
    from feature import L2_str_agg
    from collections import Counter
    
    if len(cluster_values) == 0:
        return "Empty cluster"
    
    # 获取所有值的 pattern
    patterns = [L2_str_agg(str(v)) for v in cluster_values]
    pattern_counter = Counter(patterns)
    most_common_pattern = pattern_counter.most_common(1)[0][0] if pattern_counter else "Unknown"
    
    # 基本统计
    avg_length = sum(len(str(v)) for v in cluster_values) / len(cluster_values)
    
    # 检查特殊值
    special_values = ['', 'nan', 'null', 'none', 'n/a', 'na', '-', '--']
    special_count = sum(1 for v in cluster_values if str(v).lower().strip() in special_values)
    
    if special_count > len(cluster_values) * 0.5:
        return f"Mostly empty/null values (pattern: {most_common_pattern})"
    
    # 构建描述
    description = f"Pattern: {most_common_pattern}, Avg length: {avg_length:.1f}"
    
    return description




def generate_regex_from_samples(sample_values, attr_name=None):
    """
    从样本值生成正则表达式模式
    
    Args:
        sample_values: 样本值列表
        attr_name: 属性名称（可选，用于优化）
    
    Returns:
        regex_pattern: 正则表达式字符串，如果无法生成则返回None
    """
    from feature import L2_str_agg
    from collections import Counter
    import re
    
    if not sample_values or len(sample_values) == 0:
        return None
    
    # 获取所有样本的pattern
    patterns = [L2_str_agg(str(v)) for v in sample_values]
    pattern_counter = Counter(patterns)
    
    # 如果所有样本都有相同的pattern，生成对应的正则
    if len(pattern_counter) == 1:
        pattern = list(pattern_counter.keys())[0]
        regex = pattern_to_regex(pattern, sample_values)
        return regex
    
    # 如果有多个pattern但主导pattern占比>80%，使用主导pattern
    most_common_pattern, count = pattern_counter.most_common(1)[0]
    if count / len(sample_values) >= 0.8:
        regex = pattern_to_regex(most_common_pattern, sample_values)
        return regex
    
    # 否则尝试找到更通用的模式
    # 检查是否所有值都是数字
    if all(str(v).replace('.', '').replace('-', '').replace('+', '').isdigit() for v in sample_values):
        # 数字模式
        has_decimal = any('.' in str(v) for v in sample_values)
        if has_decimal:
            return r'^\d+\.?\d*$'
        else:
            return r'^\d+$'
    
    # 检查是否都是字母
    if all(str(v).replace(' ', '').replace('-', '').isalpha() for v in sample_values):
        return r'^[A-Za-z\s\-]+$'
    
    # 无法生成统一的正则
    return None


def pattern_to_regex(pattern, sample_values):
    """
    将L2_str_agg的pattern转换为正则表达式
    
    Pattern格式: L表示字母, N表示数字, S表示符号, 用-分隔
    例如: "N-N-S" 表示 数字-数字-符号
    
    Args:
        pattern: L2_str_agg生成的pattern字符串
        sample_values: 样本值列表（用于确定具体细节）
    
    Returns:
        regex: 正则表达式字符串
    """
    import re
    
    if not pattern or pattern == "Unknown":
        return None
    
    # 分析样本值的具体特征
    sample_strs = [str(v) for v in sample_values]
    
    # 检查是否有固定的符号或单位
    # 例如: "12 oz", "16 oz" -> 固定单位 "oz"
    common_suffix = find_common_suffix(sample_strs)
    common_prefix = find_common_prefix(sample_strs)
    
    # 构建正则表达式
    regex_parts = []
    
    if common_prefix:
        regex_parts.append(re.escape(common_prefix))
    
    # 解析pattern
    parts = pattern.split('-')
    for part in parts:
        if part == 'L':
            regex_parts.append(r'[A-Za-z]+')
        elif part == 'N':
            # 检查是否有小数点
            has_decimal = any('.' in s for s in sample_strs)
            if has_decimal:
                regex_parts.append(r'\d+\.?\d*')
            else:
                regex_parts.append(r'\d+')
        elif part == 'S':
            # 符号，尝试找出具体是什么符号
            symbols = set()
            for s in sample_strs:
                for char in s:
                    if not char.isalnum() and not char.isspace():
                        symbols.add(char)
            if symbols:
                escaped_symbols = [re.escape(sym) for sym in symbols]
                regex_parts.append(f"[{''.join(escaped_symbols)}]")
            else:
                regex_parts.append(r'[^\w\s]')
        else:
            # 未知类型，使用通配
            regex_parts.append(r'.+')
    
    if common_suffix:
        regex_parts.append(re.escape(common_suffix))
    
    # 组合正则表达式
    if len(regex_parts) == 0:
        return None
    
    # 添加可能的空格
    regex = r'^\s*' + r'\s*'.join(regex_parts) + r'\s*$'
    
    return regex


def find_common_prefix(strings):
    """找到字符串列表的公共前缀"""
    if not strings:
        return ""
    
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    
    return prefix


def find_common_suffix(strings):
    """找到字符串列表的公共后缀"""
    if not strings:
        return ""
    
    # 反转字符串找前缀，然后再反转回来
    reversed_strings = [s[::-1] for s in strings]
    reversed_suffix = find_common_prefix(reversed_strings)
    
    return reversed_suffix[::-1]

def select_diverse_samples(cluster_values, max_samples=3):
    """
    从聚类值中选择尽可能不同的样本
    
    Args:
        cluster_values: 聚类中的所有值
        max_samples: 最多选择的样本数
    
    Returns:
        selected_samples: 选中的样本列表
    """
    if len(cluster_values) == 0:
        return []
    
    # 转换为字符串并去重
    unique_values = []
    seen = set()
    for val in cluster_values:
        val_str = str(val).strip()
        if val_str not in seen:
            unique_values.append(val_str)
            seen.add(val_str)
            if len(unique_values) >= max_samples:
                break
    
    # 如果唯一值少于max_samples，返回所有唯一值
    if len(unique_values) <= max_samples:
        return unique_values
    
    # 如果有足够的唯一值，选择最不同的几个
    # 简单策略：选择长度差异最大的样本
    selected = [unique_values[0]]  # 先选第一个
    
    for _ in range(max_samples - 1):
        max_diff = -1
        best_candidate = None
        
        for candidate in unique_values:
            if candidate in selected:
                continue
            
            # 计算与已选样本的最小差异
            min_diff = float('inf')
            for selected_val in selected:
                # 使用长度差异和字符差异
                len_diff = abs(len(candidate) - len(selected_val))
                char_diff = sum(1 for a, b in zip(candidate, selected_val) if a != b)
                diff = len_diff + char_diff
                min_diff = min(min_diff, diff)
            
            if min_diff > max_diff:
                max_diff = min_diff
                best_candidate = candidate
        
        if best_candidate:
            selected.append(best_candidate)
        else:
            break
    
    return selected


def get_llm_scores_for_patterns(attr_name, cluster_info_list, logger):
    """
    使用LLM比较多个聚类规范并给出分数
    
    Args:
        attr_name: 属性名称
        cluster_info_list: 列表，每个元素是 (cluster_idx, pattern_desc, sample_values, cluster_size)
        logger: 日志记录器
    
    Returns:
        scores_dict: {cluster_idx: llm_score}
    """
    if len(cluster_info_list) == 0:
        return {}
    
    # 准备数据
    patterns_with_samples = []
    cluster_indices = []
    for cluster_idx, pattern_desc, samples, cluster_size in cluster_info_list:
        patterns_with_samples.append((pattern_desc, samples, cluster_size))
        cluster_indices.append(cluster_idx)
    
    # 调用LLM
    prompt = llm_compare_patterns_canonicality_prompt(attr_name, patterns_with_samples)
    
    try:
        response = query_base(prompt)
        
        # 解析JSON响应
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response
        
        scores_data = json.loads(json_str)
        
        # 提取分数
        scores_dict = {}
        for i, cluster_idx in enumerate(cluster_indices, 1):
            pattern_key = f"pattern_{i}"
            if pattern_key in scores_data:
                score = scores_data[pattern_key].get('score', 0.5)
                reasoning = scores_data[pattern_key].get('reasoning', '')
                scores_dict[cluster_idx] = score
                logger.info(f"  聚类{cluster_idx} LLM分数: {score:.2f} - {reasoning}")
            else:
                scores_dict[cluster_idx] = 0.5
                logger.warning(f"  聚类{cluster_idx} 未找到LLM分数，使用默认值0.5")
        
        return scores_dict
        
    except Exception as e:
        logger.error(f"获取LLM规范性分数时出错: {str(e)}")
        # 返回默认分数
        return {cluster_idx: 0.5 for cluster_idx, _, _, _ in cluster_info_list}


def perform_distribution_analysis(dirty_csv, col_num, col_name, config, logger):
    """
    执行分布分析方法
    
    新逻辑：
    1. 对聚类数量>=10的聚类总结规范（最多10个，按聚类大小排序）
    2. LLM比较这些规范并给出分数
    3. 使用LLM分数计算canonical score
    4. 只保留分数高于阈值的规范
    """
    eps = config.get('eps', 0.3)
    max_cluster_centers = config.get('max_cluster_centers', 20)
    min_cluster_size_for_pattern = config.get('min_cluster_size_for_pattern', 10)
    max_patterns_to_compare = config.get('max_patterns_to_compare', 10)
    canonical_score_threshold = config.get('canonical_score_threshold', 0.5)
    alpha = config.get('alpha', 0.25)
    beta = config.get('beta', 0.15)
    gamma = config.get('gamma', 0.15)
    delta = config.get('delta', 0.45)
    
    logger.info(f"对列 '{col_name}' 执行分布分析，eps={eps}")
    
    cluster_result = single_column_dbscan_clustering(dirty_csv, col_num, col_name, eps=eps)
    
    if cluster_result is None or cluster_result['n_clusters'] == 0:
        logger.warning(f"列 '{col_name}' 聚类结果为空")
        return None
    
    logger.info(f"列 '{col_name}' 聚类完成，共 {cluster_result['n_clusters']} 个聚类")
    
    total_samples = len(dirty_csv)
    
    # 步骤1: 筛选聚类数量>=min_cluster_size_for_pattern的聚类
    large_clusters = []
    for idx, cluster_values in enumerate(cluster_result['cluster_values']):
        if len(cluster_values) >= min_cluster_size_for_pattern:
            large_clusters.append((idx, cluster_values, len(cluster_values)))
    
    logger.info(f"列 '{col_name}' 有 {len(large_clusters)} 个聚类大小>={min_cluster_size_for_pattern}")
    
    # 按聚类大小排序，取前max_patterns_to_compare个
    large_clusters.sort(key=lambda x: x[2], reverse=True)
    large_clusters = large_clusters[:max_patterns_to_compare]
    
    # 步骤2: 为这些聚类生成规范描述
    cluster_info_list = []
    for cluster_idx, cluster_values, cluster_size in large_clusters:
        pattern_desc = get_cluster_pattern_description(cluster_values)
        sample_values = select_diverse_samples(cluster_values, max_samples=3)  # 选择3个尽可能不同的样本
        cluster_info_list.append((cluster_idx, pattern_desc, sample_values, cluster_size))
        logger.info(f"  聚类{cluster_idx} (大小={cluster_size}): {pattern_desc}")
    
    # 步骤3: 使用LLM比较规范并获取分数
    llm_scores_dict = {}
    if len(cluster_info_list) > 0:
        logger.info(f"使用LLM比较 {len(cluster_info_list)} 个规范...")
        llm_scores_dict = get_llm_scores_for_patterns(col_name, cluster_info_list, logger)
    
    # 步骤4: 计算所有聚类的canonical score（使用LLM分数）
    canonical_scores = []
    score_components = []
    
    for idx, cluster_values in enumerate(cluster_result['cluster_values']):
        # 获取该聚类的LLM分数（如果有）
        llm_score = llm_scores_dict.get(idx, None)
        
        # 如果没有LLM分数（小聚类），使用默认的规范性评估
        if llm_score is None:
            # 对于小聚类，使用简单的规则评估
            special_values = ['', 'nan', 'null', 'none', 'n/a', 'na', '-', '--']
            invalid_count = sum(1 for v in cluster_values if str(v).lower().strip() in special_values)
            if invalid_count > len(cluster_values) * 0.5:
                llm_score = 0.1  # 大部分是无效值
            else:
                llm_score = 0.5  # 默认中等分数
        
        # 计算canonical score
        score, components = calculate_canonical_score(
            cluster_values, total_samples, alpha, beta, gamma, delta,
            attr_name=col_name, logger=None, use_llm_score=False  # 直接使用llm_score
        )
        
        # 手动添加LLM分数到components
        components['llm_canon'] = llm_score
        
        # 重新计算总分（包含LLM分数）
        score = alpha * components['freq'] + beta * components['reg'] + gamma * components['compact'] + delta * llm_score
        
        canonical_scores.append(score)
        score_components.append(components)
    
    # 步骤5: 只保留分数高于阈值的聚类作为canonical
    canonical_indices = []
    for idx, score in enumerate(canonical_scores):
        if score >= canonical_score_threshold:
            canonical_indices.append(idx)
    
    canonical_indices.sort(key=lambda idx: canonical_scores[idx], reverse=True)
    canonical_indices = canonical_indices[:1]   # 只保留主canonical
    
    logger.info(f"列 '{col_name}' 有 {len(canonical_indices)} 个聚类的分数>={canonical_score_threshold}")
    for i, idx in enumerate(canonical_indices[:5]):  # 只显示前5个
        logger.info(f"  Canonical {i+1}: 聚类{idx}, Score={canonical_scores[idx]:.4f}, LLM={score_components[idx]['llm_canon']:.2f}")
    
    canonical_probs = calculate_canonical_probability(canonical_scores)
    
    center_values = []
    for center_idx in cluster_result['cluster_centers'][:max_cluster_centers]:
        center_values.append(str(dirty_csv.iloc[center_idx, col_num]))
    
    analysis_result = {
        'col_name': col_name,
        'col_num': col_num,
        'n_clusters': cluster_result['n_clusters'],
        'cluster_centers': cluster_result['cluster_centers'],
        'cluster_indices': cluster_result['cluster_indices'],
        'cluster_values': cluster_result['cluster_values'],
        'center_values': center_values,
        'canonical_scores': canonical_scores,
        'score_components': score_components,
        'canonical_probs': canonical_probs,
        'top_canonical_indices': canonical_indices,  # 所有高于阈值的聚类
        'noise_indices': cluster_result['noise_indices'],
        'config': config,
        'llm_scores': llm_scores_dict
    }
    
    return analysis_result


def ask_llm_for_distribution_analysis(attr_name, center_values, logger):
    """询问LLM是否需要分布分析"""
    prompt = distribution_analysis_decision_prompt(attr_name, center_values)
    
    try:
        response = query_base(prompt)
        response = response.strip().lower()
        
        logger.info(f"列 '{attr_name}' LLM分布分析决策响应: {response}")
        
        if 'yes' in response:
            return True
        elif 'no' in response:
            return False
        else:
            logger.warning(f"列 '{attr_name}' LLM响应无法解析，默认不使用分布分析")
            return False
    except Exception as e:
        logger.error(f"询问LLM分布分析决策时出错: {str(e)}")
        return False


# 错误模式识别相关函数

def identify_error_patterns(analysis_result, dirty_csv, logger, 
                            error_pattern_threshold=0.6,
                            alpha=0.3, beta=0.3, gamma=0.4):
    """
    识别错误模式
    
    Args:
        analysis_result: 分布分析结果
        dirty_csv: 脏数据DataFrame
        logger: 日志记录器
        error_pattern_threshold: 错误模式阈值
        alpha, beta, gamma: 错误分数权重
    
    Returns:
        error_patterns: 错误模式列表
    """
    col_name = analysis_result['col_name']
    top_canonical_indices = analysis_result['top_canonical_indices']
    cluster_values = analysis_result['cluster_values']
    canonical_scores = analysis_result['canonical_scores']
    
    if len(top_canonical_indices) == 0:
        logger.warning(f"列 '{col_name}' 没有canonical模式，跳过错误模式识别")
        return []
    
    # 获取分数最高的canonical模式
    best_canonical_idx = top_canonical_indices[0]
    canonical_cluster_values = cluster_values[best_canonical_idx]
    canonical_pattern_desc = get_cluster_pattern_description(canonical_cluster_values)
    
    # 选择5个多样化的canonical样本值用于LLM评估
    canonical_samples = select_diverse_samples(canonical_cluster_values, max_samples=5)
    
    logger.info(f"列 '{col_name}' 最佳canonical模式: 聚类{best_canonical_idx}")
    logger.info(f"  描述: {canonical_pattern_desc}")
    logger.info(f"  样本值: {canonical_samples}")
    
    # 识别错误模式
    error_patterns = []
    
    # 考虑其他大聚类作为候选错误模式
    for idx, values in enumerate(cluster_values):
        # 跳过canonical模式
        if idx in top_canonical_indices:
            continue
        
        # 只考虑足够大的聚类
        if len(values) < 10:
            continue
        
        # 使用LLM判断是否为错误模式
        from prompt_gen import error_pattern_incompatibility_prompt
        error_samples = select_diverse_samples(values, max_samples=5)
        
        try:
            prompt = error_pattern_incompatibility_prompt(col_name, canonical_samples, error_samples)
            response = query_base(prompt)
            response = response.strip()
            
            # 解析LLM返回的不兼容性分数
            try:
                incompatibility_score = float(response)
                incompatibility_score = max(0.0, min(1.0, incompatibility_score))
            except ValueError:
                import re
                numbers = re.findall(r'\d+\.?\d*', response)
                if numbers:
                    incompatibility_score = float(numbers[0])
                    if incompatibility_score > 1.0:
                        incompatibility_score = incompatibility_score / 100.0
                    incompatibility_score = max(0.0, min(1.0, incompatibility_score))
                else:
                    incompatibility_score = 0.5
            
            logger.info(f"  候选错误模式 聚类{idx} (大小={len(values)}): "
                       f"不兼容性分数={incompatibility_score:.4f}")
            
            # 如果不兼容性分数超过阈值，标记为错误模式
            if incompatibility_score >= error_pattern_threshold:
                pattern_desc = get_cluster_pattern_description(values)
                
                # 生成正则表达式用于精确匹配
                regex_pattern = generate_regex_from_samples(values, col_name)
                
                error_pattern = {
                    'cluster_id': idx,
                    'pattern_description': pattern_desc,
                    'example_values': error_samples,
                    'regex_pattern': regex_pattern,
                    'cluster_size': len(values),
                    'incompatibility_score': incompatibility_score
                }
                error_patterns.append(error_pattern)
                logger.info(f"  ✓ 识别为错误模式: {pattern_desc}")
                if regex_pattern:
                    logger.info(f"    正则表达式: {regex_pattern}")
        
        except Exception as e:
            logger.warning(f"  评估聚类{idx}时出错: {str(e)}")
            continue
    
    logger.info(f"列 '{col_name}' 识别出 {len(error_patterns)} 个错误模式")
    
    return error_patterns


def check_value_matches_error_pattern(value, error_patterns):
    """
    检查值是否匹配任何错误模式（使用正则表达式精确匹配）
    
    Args:
        value: 要检查的值
        error_patterns: 错误模式列表
    
    Returns:
        matches: 是否匹配 (True/False)
        matched_pattern_idx: 匹配的模式索引 (-1表示不匹配)
    """
    import re
    
    if not error_patterns or len(error_patterns) == 0:
        return False, -1
    
    value_str = str(value).strip()
    
    for i, pattern in enumerate(error_patterns):
        regex_pattern = pattern.get('regex_pattern')
        
        # 如果有正则表达式，使用正则匹配
        if regex_pattern:
            try:
                if re.match(regex_pattern, value_str):
                    return True, i
            except re.error:
                # 正则表达式无效，跳过
                pass
        
        # 如果没有正则表达式，使用示例值精确匹配
        example_values = pattern.get('example_values', [])
        if value_str in [str(ex).strip() for ex in example_values]:
            return True, i
    
    return False, -1




def check_value_matches_canonical_pattern(value, canonical_patterns):
    """
    检查值是否匹配任何正确模式（使用正则表达式精确匹配）
    
    Args:
        value: 要检查的值
        canonical_patterns: 正确模式列表
    
    Returns:
        matches: 是否匹配 (True/False)
        matched_pattern_idx: 匹配的模式索引 (-1表示不匹配)
    """
    import re
    
    if not canonical_patterns or len(canonical_patterns) == 0:
        return False, -1
    
    value_str = str(value).strip()
    
    for i, pattern in enumerate(canonical_patterns):
        regex_pattern = pattern.get('regex_pattern')
        
        # 如果有正则表达式，使用正则匹配
        if regex_pattern and regex_pattern != 'N/A':
            try:
                if re.match(regex_pattern, value_str):
                    return True, i
            except re.error:
                # 正则表达式无效，跳过
                pass
        
        # 如果没有正则表达式，使用示例值精确匹配
        example_values = pattern.get('example_valid_values', [])
        if value_str in [str(ex).strip() for ex in example_values]:
            return True, i
    
    return False, -1

def calculate_error_pattern_feature(value, error_patterns):
    """
    计算值与错误模式的匹配特征
    
    Args:
        value: 要检查的值
        error_patterns: 错误模式列表
    
    Returns:
        feature: 特征值 (0或1)，1表示匹配错误模式
    """
    matches, pattern_idx = check_value_matches_error_pattern(value, error_patterns)
    return 1.0 if matches else 0.0



def analyze_canonical_patterns_with_llm(analysis_result, dirty_csv, logger):
    """使用LLM分析Canonical簇的标准模式"""
    col_name = analysis_result['col_name']
    top_canonical_indices = analysis_result['top_canonical_indices']
    cluster_values = analysis_result['cluster_values']
    canonical_scores = analysis_result['canonical_scores']
    score_components = analysis_result['score_components']
    llm_scores = analysis_result.get('llm_scores', {})  # 获取LLM比较分数
    max_samples = analysis_result['config'].get('max_samples_per_cluster', 10)
    
    canonical_patterns = []
    
    for idx in top_canonical_indices:
        if idx >= len(cluster_values):
            continue
        
        samples = cluster_values[idx][:max_samples]
        score = canonical_scores[idx]
        llm_canon_score = llm_scores.get(idx, score_components[idx].get('llm_canon', 0.5))
        
        prompt = canonical_pattern_analysis_prompt(col_name, samples, idx, score)
        
        try:
            response = query_base(prompt)
            
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                pattern_json = json_match.group(1)
                pattern = json.loads(pattern_json)
                pattern['cluster_id'] = idx
                pattern['canonical_score'] = score
                pattern['llm_canonicality_score'] = llm_canon_score  # 添加LLM评分
                
                # 如果LLM没有生成有效的正则表达式，自动生成一个
                if not pattern.get('regex_pattern') or pattern.get('regex_pattern') == 'N/A':
                    auto_regex = generate_regex_from_samples(cluster_values[idx], col_name)
                    if auto_regex:
                        pattern['regex_pattern'] = auto_regex
                        logger.info(f"  自动生成正则表达式: {auto_regex}")
                
                canonical_patterns.append(pattern)
                logger.info(f"列 '{col_name}' 聚类{idx} 标准模式: {pattern.get('pattern_name', 'Unknown')}, LLM分数: {llm_canon_score:.2f}")
            else:
                try:
                    pattern = json.loads(response)
                    pattern['cluster_id'] = idx
                    pattern['canonical_score'] = score
                    pattern['llm_canonicality_score'] = llm_canon_score  # 添加LLM评分
                    
                    # 如果LLM没有生成有效的正则表达式，自动生成一个
                    if not pattern.get('regex_pattern') or pattern.get('regex_pattern') == 'N/A':
                        auto_regex = generate_regex_from_samples(cluster_values[idx], col_name)
                        if auto_regex:
                            pattern['regex_pattern'] = auto_regex
                            logger.info(f"  自动生成正则表达式: {auto_regex}")
                    
                    canonical_patterns.append(pattern)
                except:
                    logger.warning(f"列 '{col_name}' 聚类{idx} 无法解析LLM响应")
                    # 自动生成正则表达式
                    auto_regex = generate_regex_from_samples(cluster_values[idx], col_name)
                    canonical_patterns.append({
                        'pattern_name': f'Pattern_{idx}',
                        'pattern_description': f'Cluster {idx} pattern',
                        'regex_pattern': auto_regex if auto_regex else 'N/A',
                        'key_characteristics': [],
                        'example_valid_values': samples[:3],
                        'common_errors': [],
                        'cluster_id': idx,
                        'canonical_score': score,
                        'llm_canonicality_score': llm_canon_score  # 添加LLM评分
                    })
                    if auto_regex:
                        logger.info(f"  自动生成正则表达式: {auto_regex}")
        except Exception as e:
            logger.error(f"分析列 '{col_name}' 聚类{idx} 标准模式时出错: {str(e)}")
            canonical_patterns.append({
                'pattern_name': f'Pattern_{idx}',
                'pattern_description': f'Cluster {idx} pattern',
                'regex_pattern': 'N/A',
                'key_characteristics': [],
                'example_valid_values': samples[:3] if samples else [],
                'common_errors': [],
                'cluster_id': idx,
                'canonical_score': score,
                'llm_canonicality_score': llm_canon_score  # 添加LLM评分
            })
    
    return canonical_patterns


def calculate_pattern_similarity_feature(value, canonical_patterns):
    """计算值与最相似标准模式的相似度特征"""
    if not canonical_patterns or len(canonical_patterns) == 0:
        return 0.0, -1
    
    value_str = str(value)
    max_similarity = 0.0
    best_pattern_idx = 0
    
    for i, pattern in enumerate(canonical_patterns):
        example_values = pattern.get('example_valid_values', [])
        if example_values:
            similarities = [calculate_string_similarity(value_str, str(ex)) for ex in example_values]
            pattern_sim = max(similarities) if similarities else 0.0
        else:
            pattern_sim = 0.0
        
        regex_pattern = pattern.get('regex_pattern', 'N/A')
        if regex_pattern and regex_pattern != 'N/A':
            try:
                if re.match(regex_pattern, value_str):
                    pattern_sim = max(pattern_sim, 0.8)
            except:
                pass
        
        if pattern_sim > max_similarity:
            max_similarity = pattern_sim
            best_pattern_idx = i
    
    return max_similarity, best_pattern_idx


def save_distribution_analysis_results(analysis_results, canonical_patterns_dict, error_patterns_dict, resp_path, logger):
    """保存分布分析结果到文件"""
    dist_analysis_dir = os.path.join(resp_path, 'distribution_analysis')
    os.makedirs(dist_analysis_dir, exist_ok=True)
    
    clustering_results = {}
    for attr, result in analysis_results.items():
        if result is None:
            continue
        clustering_results[attr] = {
            'n_clusters': result['n_clusters'],
            'cluster_centers': result['cluster_centers'],
            'cluster_indices': result['cluster_indices'],
            'center_values': result['center_values'],
            'canonical_scores': result['canonical_scores'],
            'score_components': result['score_components'],
            'canonical_probs': result['canonical_probs'],
            'top_canonical_indices': result['top_canonical_indices'],
            'noise_indices': result['noise_indices']
        }
    
    clustering_file = os.path.join(dist_analysis_dir, 'clustering_results.json')
    with open(clustering_file, 'w', encoding='utf-8') as f:
        json.dump(clustering_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    logger.info(f"聚类结果已保存到: {clustering_file}")
    
    for attr, result in analysis_results.items():
        if result is None:
            continue
        cluster_values_file = os.path.join(dist_analysis_dir, f'{attr}_cluster_values.json')
        with open(cluster_values_file, 'w', encoding='utf-8') as f:
            json.dump({
                'cluster_values': result['cluster_values'],
                'cluster_indices': result['cluster_indices']
            }, f, ensure_ascii=False, indent=2)
    
    patterns_file = os.path.join(dist_analysis_dir, 'canonical_patterns.json')
    with open(patterns_file, 'w', encoding='utf-8') as f:
        json.dump(canonical_patterns_dict, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    logger.info(f"标准模式已保存到: {patterns_file}")
    
    # 新增：保存错误模式
    error_patterns_file = os.path.join(dist_analysis_dir, 'error_patterns.json')
    with open(error_patterns_file, 'w', encoding='utf-8') as f:
        json.dump(error_patterns_dict, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    logger.info(f"错误模式已保存到: {error_patterns_file}")
    
    return dist_analysis_dir


def process_distribution_analysis_for_all_columns(dirty_csv, all_attrs, config, resp_path, logger):
    """
    对所有列执行分布分析流程
    
    修改：默认对所有列使用分布分析，不再询问LLM是否需要
    """
    distribution_analysis_results = {}
    canonical_patterns_dict = {}
    use_distribution_analysis = {}
    
    if not config.get('enabled', False):
        logger.info("分布分析方法未启用")
        for attr in all_attrs:
            use_distribution_analysis[attr] = False
        return distribution_analysis_results, canonical_patterns_dict, error_patterns_dict, use_distribution_analysis
    
    logger.info("开始分布分析流程（默认对所有列使用）...")
    
    # 获取错误模式识别配置
    error_pattern_threshold = config.get('error_pattern_threshold', 0.6)
    error_pattern_alpha = config.get('error_pattern_alpha', 0.3)
    error_pattern_beta = config.get('error_pattern_beta', 0.3)
    error_pattern_gamma = config.get('error_pattern_gamma', 0.4)
    
    error_patterns_dict = {}  # 新增：错误模式字典

    for col_num, attr in enumerate(all_attrs):
        logger.info(f"\n处理列 '{attr}' ({col_num + 1}/{len(all_attrs)})")
        
        # 执行分布分析
        analysis_result = perform_distribution_analysis(dirty_csv, col_num, attr, config, logger)
        
        if analysis_result is None:
            logger.warning(f"列 '{attr}' 分布分析失败，使用原方法")
            use_distribution_analysis[attr] = False
            continue
        
        # 默认使用分布分析（不再询问LLM）
        logger.info(f"列 '{attr}' 使用分布分析")
        use_distribution_analysis[attr] = True
        distribution_analysis_results[attr] = analysis_result
        
        # 分析canonical簇的标准模式
        canonical_patterns = analyze_canonical_patterns_with_llm(analysis_result, dirty_csv, logger)
        canonical_patterns_dict[attr] = canonical_patterns
        
        logger.info(f"列 '{attr}' 分析完成，识别出 {len(canonical_patterns)} 个标准模式")
        
        # 新增：识别错误模式
        logger.info(f"列 '{attr}' 开始识别错误模式...")
        error_patterns = identify_error_patterns(
            analysis_result, dirty_csv, logger,
            error_pattern_threshold=error_pattern_threshold,
            alpha=error_pattern_alpha,
            beta=error_pattern_beta,
            gamma=error_pattern_gamma
        )
        error_patterns_dict[attr] = error_patterns
        
        if len(error_patterns) > 0:
            logger.info(f"列 '{attr}' 识别出 {len(error_patterns)} 个错误模式")
        else:
            logger.info(f"列 '{attr}' 未识别出错误模式")
    
    # 保存结果
    if distribution_analysis_results:
        save_distribution_analysis_results(
            distribution_analysis_results, canonical_patterns_dict, error_patterns_dict, resp_path, logger
        )
    
    return distribution_analysis_results, canonical_patterns_dict, error_patterns_dict, use_distribution_analysis


# ==================== 分布分析方法相关函数结束 ====================



def convert_label_history_to_train_data(index_value_label_history, dirty_csv, related_attrs_dict, 
                                         consistency_threshold, all_attrs):
    """
    将LLM标注历史转换为训练数据
    
    Args:
        index_value_label_history: {attr: {idx: [label1, label2, ...]}}
        dirty_csv: 脏数据DataFrame
        related_attrs_dict: 相关属性字典
        consistency_threshold: 一致性阈值
        all_attrs: 所有属性列表
    
    Returns:
        train_data_dict: {attr: {'right': [(idx, value)], 'wrong': [(idx, value)]}}
        final_labels: {attr: [(idx, value, label)]} 用于评估的最终标签
    """
    train_data_dict = defaultdict(lambda: {'right': [], 'wrong': []})
    final_labels = defaultdict(list)
    
    for attr in all_attrs:
        if attr not in index_value_label_history:
            continue
            
        related_attrs = list(related_attrs_dict[attr])
        
        for idx, label_list in index_value_label_history[attr].items():
            if not label_list:
                continue
                
            consistency, majority_label = calculate_llm_consistency(label_list)
            value = dirty_csv.loc[idx, [attr] + related_attrs].to_dict()
            
            # 记录最终标签用于评估
            final_labels[attr].append((idx, value, majority_label))
            
            # 只有一致性达到阈值的样本才加入训练集
            if consistency >= consistency_threshold:
                if majority_label == 1:
                    train_data_dict[attr]['wrong'].append((idx, value))
                else:
                    train_data_dict[attr]['right'].append((idx, value))
    
    return train_data_dict, final_labels


def llm_label_indices(attr_name, indices, dirty_csv, clean_csv, related_attrs_dict, 
                      high_confidence_right_dict, high_confidence_wrong_dict,
                      error_checking_res_directory, err_check_val_num_per_query=20,
                      canonical_patterns=None, error_patterns=None):
    """
    对指定的indices进行LLM标注，累积保存标注文件，并返回当前标注结果
    
    新增：使用错误模式进行预筛选，匹配错误模式的值直接标注为错误
    
    Returns:
        current_labels: {attr: [(idx, value, label), ...]}
    """
    related_attrs = list(related_attrs_dict[attr_name])
    
    # 新增：使用错误模式进行预筛选
    pre_labeled_by_error_pattern = {}  # {idx: label} - 匹配错误模式，直接标注为错误
    indices_to_llm = []  # 需要LLM标注的索引
    
    for idx in indices:
        value = str(dirty_csv.loc[idx, attr_name])
        
        # 检查是否匹配错误模式
        if error_patterns and len(error_patterns) > 0:
            error_matches, error_pattern_idx = check_value_matches_error_pattern(value, error_patterns)
            if error_matches:
                # 匹配错误模式，直接标注为错误，不需要LLM标注
                pre_labeled_by_error_pattern[idx] = 1
                continue
        
        # 不匹配错误模式，需要LLM标注
        indices_to_llm.append(idx)
    
    # 记录预标注信息
    if len(pre_labeled_by_error_pattern) > 0:
        with open(os.path.join(error_checking_res_directory, f'error_pattern_pre_labeled_{attr_name}.txt'), 'a', encoding='utf-8') as f:
            f.write(f"// Pre-labeled by error patterns: {len(pre_labeled_by_error_pattern)} samples\n")
            for idx, label in pre_labeled_by_error_pattern.items():
                value = str(dirty_csv.loc[idx, attr_name])
                f.write(f"  idx={idx}: label={label}, value='{value}'\n")
            f.write("\n")
    
    # 将数据分成子列表进行处理
    split_indices = split_list_to_sublists(list(indices_to_llm), err_check_val_num_per_query)
    
    # 在每一批内部进行排序：匹配正确模式的放在最前面
    sorted_split_indices = []
    for batch_indices in split_indices:
        canonical_matched = []
        other_matched = []
        
        for idx in batch_indices:
            value = str(dirty_csv.loc[idx, attr_name])
            # 检查是否匹配正确模式
            if canonical_patterns and len(canonical_patterns) > 0:
                canonical_matches, _ = check_value_matches_canonical_pattern(value, canonical_patterns)
                if canonical_matches:
                    canonical_matched.append(idx)
                else:
                    other_matched.append(idx)
            else:
                other_matched.append(idx)
        
        # 合并：正确模式匹配的放在最前面
        sorted_batch = canonical_matched + other_matched
        sorted_split_indices.append(sorted_batch)
        
        # 记录每批中匹配正确模式的数量
        if len(canonical_matched) > 0:
            with open(os.path.join(error_checking_res_directory, f'canonical_pattern_matched_{attr_name}.txt'), 'a', encoding='utf-8') as f:
                f.write(f"// Batch with {len(canonical_matched)} canonical matches (prioritized):\n")
                for idx in canonical_matched:
                    value = str(dirty_csv.loc[idx, attr_name])
                    f.write(f"  idx={idx}: value='{value}'\n")
                f.write("\n")
    
    # 为排序后的索引创建数据字典
    split_values = []
    for batch_indices in sorted_split_indices:
        batch_values = ["{" + ",".join(f'"{col}":"{dirty_csv.loc[idx, col]}"' for col in [attr_name] + related_attrs) + "}" for idx in batch_indices]
        split_values.append(batch_values)
    
    split_indices = sorted_split_indices
    
    all_responses = []
    
    all_responses = []
    
    for sub_list_values, sub_list_indices in zip(split_values, split_indices):
        try:
            vals_str = '\n'.join(sub_list_values)
            # 根据是否有标准模式选择不同的prompt
            if canonical_patterns and len(canonical_patterns) > 0:
                prompt = error_check_with_canonical_prompt(
                    vals_str, attr_name, high_confidence_right_dict, 
                    high_confidence_wrong_dict, canonical_patterns
                )
            else:
                prompt = error_check_prompt(vals_str, attr_name, high_confidence_right_dict, high_confidence_wrong_dict)
            
            response = query_base(prompt)
            response = fix_error_flags(response)
            
            with open(os.path.join(error_checking_res_directory, f'prompt_error_checking_{attr_name}.txt'), 'a', encoding='utf-8') as f:
                f.write(prompt + '\n\n')
            
            with open(os.path.join(error_checking_res_directory, f'error_checking_{attr_name}.txt'), 'a', encoding='utf-8') as f:
                f.write(f"// indices: {sub_list_indices}\n")
                f.write(response + '\n')
                
                # 添加Ground Truth对比
                f.write("\n// ========== Ground Truth Comparison ==========\n")
                correct_count = 0
                total_count = 0
                
                # 从response中提取LLM标注
                llm_labels = {}
                full_pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"([^"]*)",\s*\n\s*"has_error_in_{attr_name}_value":\s*(true|false)'
                for m in re.finditer(full_pattern, response, re.IGNORECASE):
                    value_row = m.group(1)
                    has_error = m.group(3).lower() == 'true'
                    llm_labels[value_row] = 1 if has_error else 0
                
                for idx in sub_list_indices:
                    dirty_val = str(dirty_csv.loc[idx, attr_name])
                    clean_val = str(clean_csv.loc[idx, attr_name])
                    is_actually_wrong = (dirty_val != clean_val)
                    
                    # 构造value_row用于匹配
                    related_attrs_local = list(related_attrs_dict[attr_name])
                    value_dict = {col: str(dirty_csv.loc[idx, col]) for col in [attr_name] + related_attrs_local}
                    value_row_str = "{" + ",".join(f'"{k}":"{v}"' for k, v in value_dict.items()) + "}"
                    
                    # 尝试多种匹配方式
                    llm_predicted_wrong = None
                    for vr_key in llm_labels.keys():
                        if attr_name in vr_key and dirty_val in vr_key:
                            llm_predicted_wrong = llm_labels[vr_key]
                            break
                    
                    if llm_predicted_wrong is None:
                        llm_predicted_wrong = 0  # 默认预测为正确
                    
                    is_correct = (llm_predicted_wrong == 1 and is_actually_wrong) or (llm_predicted_wrong == 0 and not is_actually_wrong)
                    if is_correct:
                        correct_count += 1
                    total_count += 1
                    
                    status = "✓ CORRECT" if is_correct else "✗ WRONG"
                    f.write(f"  idx={idx}: {status}\n")
                    f.write(f"    Dirty Value:  '{dirty_val}'\n")
                    f.write(f"    Clean Value:  '{clean_val}'\n")
                    f.write(f"    Actually Wrong: {is_actually_wrong}\n")
                    f.write(f"    LLM Predicted Wrong: {llm_predicted_wrong == 1}\n")
                    f.write("\n")
                
                accuracy = correct_count / total_count if total_count > 0 else 0
                f.write(f"// Batch Accuracy: {correct_count}/{total_count} = {accuracy:.4f}\n")
                f.write("="*80 + "\n\n")
            
            all_responses.append((response, sub_list_indices))
            
        except Exception as e:
            print(f"处理属性 {attr_name} 的子任务时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    current_labels = extract_labels_from_responses(attr_name, all_responses, dirty_csv, related_attrs_dict)
    
    # 合并错误模式预标注的结果
    if len(pre_labeled_by_error_pattern) > 0:
        if attr_name not in current_labels:
            current_labels[attr_name] = []
        
        for idx, label in pre_labeled_by_error_pattern.items():
            value = dirty_csv.loc[idx, [attr_name] + related_attrs].to_dict()
            current_labels[attr_name].append((idx, value, label))
    
    return current_labels


def extract_labels_from_responses(attr_name, responses_with_indices, dirty_csv, related_attrs_dict):
    """从LLM响应中提取标注结果"""
    index_value_label_dict = defaultdict(list)
    related_attrs = list(related_attrs_dict[attr_name])
    
    # 需要过滤的关键词列表（如果error_analysis包含这些词且标记为错误，则改为正确）
    filter_keywords = ['duplicate', 'duplication', 'type']
    
    for response, indices in responses_with_indices:
        resp_content = response.replace('\\+', '').replace('\\n', '\n')
        
        wrong_pattern = err_pat_in_text_attr(attr_name)
        right_pattern = right_pat_in_text_attr(attr_name)
        
        # 新增：提取带有error_analysis的完整模式，用于检查是否需要过滤
        # 匹配格式: "value_row": "...", "error_analysis": "...", "has_error_in_xxx_value": true/false
        full_pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"([^"]*)",\s*\n\s*"has_error_in_{attr_name}_value":\s*(true|false)'
        
        events = []
        
        # 使用完整模式提取，同时检查error_analysis内容
        for m in re.finditer(full_pattern, resp_content, re.IGNORECASE):
            value_row = m.group(1)
            error_analysis = m.group(2).lower()
            has_error = m.group(3).lower() == 'true'
            
            text = normalize_string(
                value_row.replace("':'", "': '").replace(',', ', ').replace(',  ', ', ').replace('"', "'")
            ).replace('"{', '{', 1)[:-1] if value_row.startswith('"{') else normalize_string(value_row)
            
            # 检查是否需要过滤：如果error_analysis包含过滤关键词且标记为错误，则改为正确
            if has_error and any(keyword in error_analysis for keyword in filter_keywords):
                # 将错误标记改为正确
                status = 0
            else:
                status = 1 if has_error else 0
            
            events.append((m.start(), text, status))
        
        events.sort(key=lambda x: x[0])
        
        value_status = {}
        for _, value_str, status in events:
            value_status[value_str] = status
        
        for idx in indices:
            value = dirty_csv.loc[idx, [attr_name] + related_attrs].to_dict()
            norm_value = normalize_string(str(value))
            status = value_status.get(norm_value, 0)
            index_value_label_dict[attr_name].append((idx, value, status))
    
    return index_value_label_dict


def extract_func(text_content):
    # 确保text_content是字符串类型
    if text_content is None:
        return [], []
    if not isinstance(text_content, str):
        try:
            text_content = str(text_content)
        except Exception as e:
            print(f"Cannot convert text_content to string: {e}")
            return [], []
    
    try:
        code_blocks = re.findall(r'```(.*?)```', text_content, re.DOTALL)
    except (re.error, TypeError) as e:
        print(f"Regex error: {e}")
        return [], []
    clean_func_list = []
    dirty_func_list = []
    for code_block in code_blocks:
        functions = re.findall(r'def \w+\(.*?\):\n(?:[ \t]*\n)*(?: .*\n)+', code_block)
        for function in functions:
            try:
                function_name = re.findall(r'def (\w+)', function)[0]
            except IndexError:
                continue
            if 'is_clean' in function_name:
                clean_func_list.append(function)
            elif 'is_dirty' in function_name:
                dirty_func_list.append(function)
    return clean_func_list, dirty_func_list


funcs_with_errors = set()

def fix_error_flags(response_str):
    lines = response_str.splitlines()
    fixed_lines = lines.copy()
    for i in range(len(lines) - 1):
        line1 = lines[i]
        line2 = lines[i + 1]
        if '"error_analysis"' in line1 and re.search(r'not match|duplicate', line1, re.IGNORECASE):
            if re.search(r'"has_error_in_[^"]+"\s*:\s*true', line2, re.IGNORECASE):
                fixed_lines[i + 1] = re.sub(r'\btrue\b', 'false', line2, flags=re.IGNORECASE)
    return "\n".join(fixed_lines)


def normalize_string(s):
    return str(s.replace(" \\", "\\")
               .replace("\\\\", "\\")
               .replace("\\", "")
               .replace(", ", ",")
               .replace(": ", ":")
               .replace("'", '"'))


# 简化版的特征生成函数（不使用预函数和函数特征）

def process_attr_train_feat_simplified(attr, dirty_csv, train_data_dict, related_attrs_dict, 
                                       resp_path, canonical_patterns=None, error_patterns=None):
    """
    处理属性的训练特征（简化版，不使用函数特征）
    
    特征包括：
    1. FastText词向量
    2. 标准模式相似度（如果有）
    """
    fasttext_model = fasttext.load_model('./cc.en.300.bin')
    fasttext_dimension = len(dirty_csv.columns)
    fasttext.util.reduce_model(fasttext_model, fasttext_dimension)
    
    feature_list = []
    label_list = []
    related_attrs = list(related_attrs_dict[attr])
    
    # 从train_data_dict获取训练数据
    right_samples = train_data_dict.get(attr, {}).get('right', [])
    wrong_samples = train_data_dict.get(attr, {}).get('wrong', [])
    
    for idx, val in tqdm(right_samples, ncols=120, desc=f"Processing {attr} right values"):
        feature = single_val_feat_simplified(val, fasttext_model, attr, 
                                            list(dirty_csv.columns), canonical_patterns, error_patterns)
        if feature:
            feature_list.append(feature)
            label_list.append(0)
    
    for idx, val in tqdm(wrong_samples, ncols=120, desc=f"Processing {attr} wrong values"):
        feature = single_val_feat_simplified(val, fasttext_model, attr, 
                                            list(dirty_csv.columns), canonical_patterns, error_patterns)
        if feature:
            feature_list.append(feature)
            label_list.append(1)
    
    return attr, feature_list, label_list


def single_val_feat_simplified(val, fasttext_m, attr, all_attrs, canonical_patterns=None, error_patterns=None):
    """
    简化版的单值特征生成（不使用函数特征）
    
    特征包括：
    1. FastText词向量
    2. 标准模式相似度（如果有）
    """
    feature = []
    
    # 1. FastText词向量特征
    if fasttext_m is not None:
        if isinstance(val, dict):
            for a_val in val.values():
                feature.extend(fasttext_m.get_word_vector(str(a_val)))
        else:
            for a_val in val:
                feature.extend(fasttext_m.get_word_vector(str(a_val)))
    
    # 2. 添加标准模式相似度特征（如果有）
    if canonical_patterns and len(canonical_patterns) > 0:
        # 获取当前列的值
        if isinstance(val, dict):
            attr_val = val.get(attr, '')
        else:
            attr_val = str(val)
        pattern_sim, _ = calculate_pattern_similarity_feature(attr_val, canonical_patterns)
        feature.append(pattern_sim)
    
    # 3. 添加错误模式相似度特征（如果有）
    if error_patterns and len(error_patterns) > 0:
        # 获取当前列的值
        if isinstance(val, dict):
            attr_val = val.get(attr, '')
        else:
            attr_val = str(val)
        error_pattern_sim = calculate_error_pattern_feature(attr_val, error_patterns)
        feature.append(error_pattern_sim)
    
    return feature

def calculate_llm_consistency(label_history):
    """
    计算LLM标注一致性分数
    
    Args:
        label_history: 标签历史列表 [label1, label2, ...]
    
    Returns:
        consistency_score: 一致性分数 (0-1)
        majority_label: 多数标签
    """
    if not label_history or len(label_history) == 0:
        return 0.0, 0
    
    counter = Counter(label_history)
    majority_label, majority_count = counter.most_common(1)[0]
    consistency_score = majority_count / len(label_history)
    
    return consistency_score, majority_label

def make_predictions_simplified(col, attr, dirty_csv, model_col, related_attrs_dict, 
                                resp_path, canonical_patterns=None, error_patterns=None):
    """
    简化版的预测函数（不使用函数特征）
    """
    if attr not in model_col.keys():
        return []
    
    model = model_col[attr]
    related_attrs = list(related_attrs_dict[attr])
    columns = list(dirty_csv.columns)
    
    # 加载FastText模型
    fasttext_model = fasttext.load_model('./cc.en.300.bin')
    fasttext_dimension = len(columns)
    fasttext.util.reduce_model(fasttext_model, fasttext_dimension)
    
    # 生成特征
    feature_list = []
    for idx in range(len(dirty_csv)):
        cell_val = dirty_csv.loc[idx, [attr]+related_attrs].to_dict()
        feature = single_val_feat_simplified(cell_val, fasttext_model, attr, 
                                            columns, canonical_patterns, error_patterns)
        feature_list.append(feature)
    
    # 预测
    test_feat_np = np.array(feature_list)
    pred_prob_list = model.predict(test_feat_np)
    
    wrong_cells = []
    for idx in range(len(dirty_csv)):
        pred_prob = pred_prob_list[idx]
        if pred_prob == 1:
            wrong_cells.append((idx, attr))
    
    return wrong_cells



def process_attr_train_feat(attr, dirty_csv, train_data_dict, related_attrs_dict, 
                            funcs_for_attr, feature_all_dict, resp_path, canonical_patterns=None):
    """处理属性的训练特征"""
    fasttext_model = fasttext.load_model('./cc.en.300.bin')
    fasttext_dimension = len(dirty_csv.columns)
    fasttext.util.reduce_model(fasttext_model, fasttext_dimension)
    
    feature_list = []
    label_list = []
    related_attrs = list(related_attrs_dict[attr])
    
    # 从train_data_dict获取训练数据
    right_samples = train_data_dict.get(attr, {}).get('right', [])
    wrong_samples = train_data_dict.get(attr, {}).get('wrong', [])
    
    for idx, val in tqdm(right_samples, ncols=120, desc=f"Processing {attr} right values"):
        feature = single_val_feat(val, fasttext_model, funcs_for_attr, attr, -1, 
                                  list(dirty_csv.columns), feature_all_dict, resp_path,
                                  canonical_patterns=canonical_patterns)
        if feature:
            feature_list.append(feature)
            label_list.append(0)
    
    for idx, val in tqdm(wrong_samples, ncols=120, desc=f"Processing {attr} wrong values"):
        feature = single_val_feat(val, fasttext_model, funcs_for_attr, attr, -1, 
                                  list(dirty_csv.columns), feature_all_dict, resp_path,
                                  canonical_patterns=canonical_patterns)
        if feature:
            feature_list.append(feature)
            label_list.append(1)
    
    return attr, feature_list, label_list


def single_val_feat(val, fasttext_m, funcs_for_attr, attr, idx, all_attrs, feature_all_dict, resp_path, canonical_patterns=None):
    feature = []
    
    # 函数特征
    if attr in funcs_for_attr and 'clean' in funcs_for_attr[attr]:
        for func in funcs_for_attr[attr]['clean']:
            result = handle_func_exec(func, val, attr)
            feature.append(result if result != -1 else 0)
    
    if idx == -1:
        # 训练时使用fasttext
        if fasttext_m is not None:
            if isinstance(val, dict):
                for a_val in val.values():
                    feature.extend(fasttext_m.get_word_vector(str(a_val)))
            else:
                for a_val in val:
                    feature.extend(fasttext_m.get_word_vector(str(a_val)))
        # 添加标准模式相似度特征（如果有）
        if canonical_patterns and len(canonical_patterns) > 0:
            # 获取当前列的值
            if isinstance(val, dict):
                attr_val = val.get(attr, '')
            else:
                attr_val = str(val)
            pattern_sim, _ = calculate_pattern_similarity_feature(attr_val, canonical_patterns)
            feature.append(pattern_sim)
        
        return feature
    else:
        # 预测时从缓存获取
        if feature_all_dict is not None:
            fasttext_feat = feature_all_dict.get((idx, all_attrs.index(attr)), {}).get('fasttext_feat', [])
            if len(fasttext_feat) == 0 or len(fasttext_feat) < len(all_attrs):
                fasttext_feat = []
                fasttext_m = fasttext.load_model('./cc.en.300.bin')
                fasttext_dimension = len(all_attrs)
                fasttext.util.reduce_model(fasttext_m, fasttext_dimension)
                if isinstance(val, np.ndarray):
                    for a_val in val:
                        fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
                elif isinstance(val, dict):
                    for a_val in val.values():
                        fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
                else:
                    for a_val in val:
                        fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
            feature.extend(fasttext_feat)
        else:
            fasttext_m = fasttext.load_model('./cc.en.300.bin')
            fasttext_dimension = len(all_attrs)
            fasttext.util.reduce_model(fasttext_m, fasttext_dimension)
            fasttext_feat = []
            if isinstance(val, np.ndarray):
                for a_val in val:
                    fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
            elif isinstance(val, dict):
                for a_val in val.values():
                    fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
            else:
                for a_val in val:
                    fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
            feature.extend(fasttext_feat)
        
        # 添加标准模式相似度特征（如果有）
        if canonical_patterns and len(canonical_patterns) > 0:
            if isinstance(val, dict):
                attr_val = val.get(attr, '')
            else:
                attr_val = str(val)
            pattern_sim, _ = calculate_pattern_similarity_feature(attr_val, canonical_patterns)
            feature.append(pattern_sim)
        
        return idx, feature


def make_predictions(col, attr, dirty_csv, model_col, related_attrs_dict, funcs_for_attr, feature_all_dict, resp_path, canonical_patterns=None):
    if attr not in model_col.keys():
        return []
    model = model_col[attr]
    related_attrs = list(related_attrs_dict[attr])
    columns = list(dirty_csv.columns)
    
    results = []
    for idx in range(len(dirty_csv)):
        cell_val = dirty_csv.loc[idx, [attr]+related_attrs].to_dict()
        result = single_val_feat(cell_val, None, funcs_for_attr, attr, idx, columns, feature_all_dict, resp_path, canonical_patterns=canonical_patterns)
        results.append(result)
    
    sorted_results = sorted([(r[0], r[1]) for r in results])
    test_feat_list = [feat for idx, feat in sorted_results]
    
    test_feat_np = np.array(test_feat_list)
    pred_prob_list = model.predict(test_feat_np)
    wrong_cells = []
    for idx, cell_val in dirty_csv.iloc[:, col].items():
        pred_prob = pred_prob_list[idx]
        if pred_prob == 1:
            wrong_cells.append((idx, attr))
    return wrong_cells


def train_model(attr, feature_list, label_list, num_epochs):
    if feature_list is None or len(feature_list) == 0:
        return attr, None, 'mlp', 'optimizer', "None", 500
    if len(feature_list[0]) == 0:
        return attr, None, 'mlp', 'optimizer', "None", 500
    
    # 检查是否有两个类别，如果只有一个类别，使用DummyClassifier
    unique_labels = set(label_list)
    if len(unique_labels) < 2:
        from sklearn.dummy import DummyClassifier
        feat_np = np.array(feature_list)
        label_np = np.array(label_list)
        model = DummyClassifier(strategy='most_frequent')
        model.fit(feat_np, label_np)
        return attr, model, 'dummy', 'optimizer', model, num_epochs
    
    feat_np = np.array(feature_list)
    label_np = np.array(label_list)
    
    input_dim = feat_np.shape[1]
    
    model = MLPClassifier(
        hidden_layer_sizes=(2 * input_dim, input_dim),
        activation='relu',
        solver='adam',
        max_iter=num_epochs,
        random_state=42,
        n_iter_no_change=10,
        verbose=False
    )
    
    model.fit(feat_np, label_np)
    return attr, model, 'mlp', 'optimizer', model, num_epochs


def process_related_attr(RELATED_ATTRS, REL_TOP, resp_path, clean_csv, dirty_csv, all_attrs):
    related_attrs_dict = {}
    gt_wrong_dict = {}
    if RELATED_ATTRS:
        nmi_results = cal_all_column_nmi(dirty_csv)
        related_attrs_dict = cal_strong_res_column_nmi(nmi_results, rel_top=REL_TOP)
        with open(os.path.join(resp_path, 'related_attrs_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(related_attrs_dict, f, ensure_ascii=False, indent=4)
    else:
        for attr in all_attrs:
            related_attrs_dict[attr] = []
        with open(os.path.join(resp_path, 'related_attrs_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(related_attrs_dict, f, ensure_ascii=False, indent=4)

    for attr in all_attrs:
        related_attrs = list(related_attrs_dict[attr])
        if attr not in gt_wrong_dict:
            gt_wrong_dict[attr] = set()
        for i in range(len(dirty_csv)):
            if str(dirty_csv.loc[i, attr]) != str(clean_csv.loc[i, attr]) or str(clean_csv.loc[i, attr]) == 'nan':
                wrong_tuple = str(dirty_csv.loc[i, [attr] + related_attrs].to_dict())
                gt_wrong_dict[attr].add(wrong_tuple)
    return related_attrs_dict, gt_wrong_dict


def process_cluster(CLUSTER_RATE, dataset, resp_path, dirty_csv, all_attrs, related_attrs_dict, pre_funcs_for_attr):
    cluster_index_dict = {}
    center_value_dict = {}
    feature_all_dict = defaultdict(default_dict_of_lists)
    
    for col in range(len(all_attrs)):
        try:
            col_result, center_list, cluster_list, val_feat_dict, feature_dict_attr = cluster(
                dataset, 'KMeans', CLUSTER_RATE, col, related_attrs_dict, pre_funcs_for_attr, resp_path
            )
            cluster_list.insert(0, center_list)
            cluster_index_dict[all_attrs[col]] = cluster_list
            feature_all_dict.update(feature_dict_attr)
        except Exception as e:
            print(f"列 {col} ({all_attrs[col]}) 处理出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    for key, value in cluster_index_dict.items():
        temp_list = []
        related_attrs = list(related_attrs_dict[key])
        for ind in value[0]:
            temp_list.append(dirty_csv.loc[ind, [key] + related_attrs].to_dict())
        center_value_dict[key] = temp_list
    
    with open(os.path.join(resp_path, 'center_value_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(center_value_dict, f, ensure_ascii=False, indent=4)
    
    serializable_cluster_index_dict = {
        attr: [[int(idx) for idx in cluster] for cluster in clusters]
        for attr, clusters in cluster_index_dict.items()
    }
    with open(os.path.join(resp_path, 'cluster_index_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(serializable_cluster_index_dict, f, ensure_ascii=False, indent=4)
    
    with open(os.path.join(resp_path, 'cluster_feat_dict.pkl'), 'wb') as f:
        pickle.dump(feature_all_dict, f)
    
    return cluster_index_dict, center_value_dict, feature_all_dict


def calculate_jsd(p, q):
    """计算Jensen-Shannon Divergence"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.sum(p)
    q = q / np.sum(q)
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


def calculate_ksd(sample1, sample2):
    """计算Kolmogorov-Smirnov Distance"""
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    ks_statistic, p_value = stats.ks_2samp(sample1, sample2)
    return ks_statistic


def err_pat_in_text_attr(attr):
    pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"[^"]*",\s*\n\s*"has_error_in_{attr}_value":\s*true'
    return pattern


def right_pat_in_text_attr(attr):
    pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"[^"]*",\s*\n\s*"has_error_in_{attr}_value":\s*false'
    return pattern


def save_label_dict(index_value_label_dict, save_path):
    """将标注结果保存到文件"""
    with open(save_path, 'a', encoding='utf-8') as f:
        for attr, items in index_value_label_dict.items():
            for idx, value, label in items:
                rec = {"attr": attr, "idx": int(idx), "value": value, "label": label}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_mlp_prediction_errors(dirty_csv, clean_csv, det_wrong_list_res, all_attrs, resp_path):
    """
    保存MLP预测错误的值
    
    包括：
    1. False Positives: 预测为错误但实际正确
    2. False Negatives: 预测为正确但实际错误
    """
    mlp_errors_file = os.path.join(resp_path, 'mlp_prediction_errors.txt')
    
    with open(mlp_errors_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MLP PREDICTION ERRORS ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # False Positives: 预测错误但实际正确
        f.write("1. FALSE POSITIVES (Predicted as Error but Actually Clean):\n")
        f.write("-"*80 + "\n")
        fp_count = 0
        for idx, attr in det_wrong_list_res:
            dirty_val = str(dirty_csv.loc[idx, attr])
            clean_val = str(clean_csv.loc[idx, attr])
            if dirty_val == clean_val:
                fp_count += 1
                f.write(f"  Row {idx}, Column '{attr}':\n")
                f.write(f"    Dirty Value:  '{dirty_val}'\n")
                f.write(f"    Clean Value:  '{clean_val}'\n")
                f.write(f"    Status: Same (False Positive)\n\n")
        
        f.write(f"Total False Positives: {fp_count}\n\n")
        
        # False Negatives: 预测正确但实际错误
        f.write("2. FALSE NEGATIVES (Predicted as Clean but Actually Error):\n")
        f.write("-"*80 + "\n")
        fn_count = 0
        detected_set = set(det_wrong_list_res)
        
        for attr in all_attrs:
            for idx in range(len(dirty_csv)):
                dirty_val = str(dirty_csv.loc[idx, attr])
                clean_val = str(clean_csv.loc[idx, attr])
                if dirty_val != clean_val and (idx, attr) not in detected_set:
                    fn_count += 1
                    f.write(f"  Row {idx}, Column '{attr}':\n")
                    f.write(f"    Dirty Value:  '{dirty_val}'\n")
                    f.write(f"    Clean Value:  '{clean_val}'\n")
                    f.write(f"    Status: Different (False Negative)\n\n")
        
        f.write(f"Total False Negatives: {fn_count}\n\n")
        
        # 统计信息
        f.write("="*80 + "\n")
        f.write("SUMMARY:\n")
        f.write(f"  False Positives: {fp_count}\n")
        f.write(f"  False Negatives: {fn_count}\n")
        f.write(f"  Total Prediction Errors: {fp_count + fn_count}\n")
        f.write("="*80 + "\n")
    
    return mlp_errors_file


def print_prediction_errors(dirty_csv, clean_csv, det_wrong_list_res, all_attrs, related_attrs_dict, logger, resp_path):
    """打印预测错误的数据"""
    false_positives = []
    for idx, attr in det_wrong_list_res:
        if str(dirty_csv.loc[idx, attr]) == str(clean_csv.loc[idx, attr]):
            false_positives.append((idx, attr))
    
    false_negatives = []
    for attr in all_attrs:
        for idx in range(len(dirty_csv)):
            if str(dirty_csv.loc[idx, attr]) != str(clean_csv.loc[idx, attr]):
                if (idx, attr) not in det_wrong_list_res:
                    false_negatives.append((idx, attr))
    
    logger.info(f"\n误报数据: {len(false_positives)} 个")
    logger.info(f"漏报数据: {len(false_negatives)} 个")
    
    total_errors = sum(1 for attr in all_attrs for idx in range(len(dirty_csv)) 
                       if str(dirty_csv.loc[idx, attr]) != str(clean_csv.loc[idx, attr]))
    
    detected_errors = len(det_wrong_list_res)
    precision = (detected_errors - len(false_positives)) / detected_errors if detected_errors else 0
    recall = (detected_errors - len(false_positives)) / total_errors if total_errors else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    error_detail_file = os.path.join(resp_path, "prediction_error_details.txt")
    with open(error_detail_file, 'w', encoding='utf-8') as f:
        f.write("="*50 + " 预测错误数据分析 " + "="*50 + "\n\n")
        f.write(f"总错误数: {total_errors}\n")
        f.write(f"检测到的错误数: {detected_errors}\n")
        f.write(f"误报数: {len(false_positives)}\n")
        f.write(f"漏报数: {len(false_negatives)}\n")
        f.write(f"精确率: {precision:.4f}\n")
        f.write(f"召回率: {recall:.4f}\n")
        f.write(f"F1分数: {f1_score:.4f}\n")
    
    logger.info(f"详细错误信息已保存到: {error_detail_file}")


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='run_config.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Model settings
    MODEL_TYPE = config['model']['model_type']
    CLUSTER_RATE = config['model']['cluster_rate']
    API_USE = config['model']['api_use']
    RELATED_ATTRS = config['model']['related_attrs']
    PRE_FUNC_USE = config['model']['pre_func_use']
    FUNC_USE = config['model']['func_use']
    REL_TOP = config['model']['rel_top']
    LABEL_PROP = config['model']['label_prop']
    ITERATIONS = config['model']['iterations']
    INITIAL_LLM_LABEL_ITERATIONS = config['model']['initial_llm_label_iterations']
    INITIAL_LLM_LABEL_CONSISTENCY_THRESHOLD = config['model']['initial_llm_label_consistency_threshold']
    TRAIN_HIGH_CONFIDENCE_THRESHOLD = config['model']['train_high_confidence_threshold']
    MID_CONFIDENCE_THRESHOLD = config['model']['mid_confidence_threshold']
    HIGH_CONFIDENCE_THRESHOLD = config['model']['high_confidence_threshold']
    CLUSTER_SELECTION_WINDOW = config['model'].get('cluster_selection_window', -1)
    COMPUTE_F1_PER_ITERATION = config['model'].get('compute_f1_per_iteration', False)
    RESULT_ANALYZE = config['model'].get('result_analyze', False)

    
    # 分布分析配置
    DISTRIBUTION_ANALYSIS_CONFIG = config['model'].get('distribution_analysis', {
        'enabled': False,
        'eps': 0.3,
        'max_cluster_centers': 20,
        'top_canonical_clusters': 2,
        'max_samples_per_cluster': 10,
        'alpha': 0.4,
        'beta': 0.3,
        'gamma': 0.3
    })
    # Dataset settings
    base_dir = config['data']['base_dir']
    err_rate_list = config['data']['err_rate_list']
    all_set_num = config['data']['all_set_num']
    dataset_list = config['data']['datasets'] * all_set_num
    result_dir = config['data']['result_dir']
    dataset_list = sorted(dataset_list)
    set_num_list = [i % all_set_num + 1 for i in range(len(dataset_list))]
    err_check_val_num_per_query = config['data']['err_check_val_num_per_query']
    
    for set_num, dataset in zip(set_num_list, dataset_list):
        for err_rate in err_rate_list:
            date_time = datetime.now().strftime("%m-%d")
            now_time = datetime.now().strftime("%H-%M")  # 使用 - 替代 : 以兼容 Windows
            resp_path = f"{base_dir}/result/{result_dir}/{MODEL_TYPE} {date_time} {now_time} {dataset}{err_rate}-set{set_num}-iterations{ITERATIONS}"
            error_checking_res_directory = f'{resp_path}/error_checking'
            funcs_directory = f'{resp_path}/funcs'
            funcs_pre_directory = f'{resp_path}/funcs_pre'
            os.makedirs(resp_path, exist_ok=True)
            os.makedirs(error_checking_res_directory, exist_ok=True)
            os.makedirs(funcs_directory, exist_ok=True)
            os.makedirs(funcs_pre_directory, exist_ok=True)
            
            # 读取数据
            dirty_path = base_dir + '/data/' + dataset + '_error-' + str(err_rate) + '.csv'
            clean_path = base_dir + '/data/' + dataset + '_clean.csv'
            clean_csv = pd.read_csv(clean_path, dtype=str).fillna('nan')
            dirty_csv = pd.read_csv(dirty_path, dtype=str).fillna('nan')
            all_attrs = list(dirty_csv.columns)
            
            # 初始化日志和文件
            logger = Logger(resp_path)
            time_file = open(os.path.join(resp_path, '0-time.txt'), 'w', encoding='utf-8')
            para_file = open(os.path.join(resp_path, '0-para.txt'), 'w', encoding='utf-8')
            para_file.write(f"Config: {args.config}\n")
            para_file.write(f"Dataset: {dataset}, Error Rate: {err_rate}\n")
            para_file.write(f"Iterations: {ITERATIONS}, Initial LLM Iterations: {INITIAL_LLM_LABEL_ITERATIONS}\n")
            
            total_time = 0
            time_start = time.time()
            
# 这是简化后的主流程代码片段

# 在 if __name__ == "__main__": 部分的主循环中

            # ==================== 步骤1: 计算相关属性 ====================
            related_attrs_dict, gt_wrong_dict = {}, {}
            with Timer('Getting Related Attributes', logger, time_file) as t:
                related_attrs_dict, gt_wrong_dict = process_related_attr(
                    RELATED_ATTRS, REL_TOP, resp_path, clean_csv, dirty_csv, all_attrs
                )
            total_time += t.duration

            # ==================== 步骤2: 分布分析（可选） ====================
            distribution_analysis_results = {}
            canonical_patterns_dict = {}
            use_distribution_analysis = {}
            
            if DISTRIBUTION_ANALYSIS_CONFIG.get('enabled', False):
                logger.info("分布分析方法已启用")
                with Timer('Distribution Analysis', logger, time_file) as t:
                    distribution_analysis_results, canonical_patterns_dict, error_patterns_dict, use_distribution_analysis = \
                        process_distribution_analysis_for_all_columns(
                            dirty_csv, all_attrs, DISTRIBUTION_ANALYSIS_CONFIG, resp_path, logger
                        )
                total_time += t.duration
                
                # 记录哪些列使用了分布分析
                dist_analysis_attrs = [attr for attr, use in use_distribution_analysis.items() if use]
                logger.info(f"使用分布分析的列: {dist_analysis_attrs}")
                para_file.write(f"Distribution analysis enabled for: {dist_analysis_attrs}\n")
                
                # 统计错误模式
                total_error_patterns = sum(len(patterns) for patterns in error_patterns_dict.values())
                logger.info(f"共识别出 {total_error_patterns} 个错误模式")
                para_file.write(f"Total error patterns identified: {total_error_patterns}\n")
            else:
                logger.info("分布分析方法未启用")
                error_patterns_dict = {}
                for attr in all_attrs:
                    use_distribution_analysis[attr] = False
                    error_patterns_dict[attr] = []
            
            # ==================== 步骤3: 聚类 ====================
            cluster_index_dict, center_value_dict = {}, {}
            feature_all_dict = defaultdict(default_dict_of_lists)
            with Timer('Clustering', logger, time_file) as t:
                cluster_index_dict, center_value_dict, feature_all_dict = process_cluster(
                    CLUSTER_RATE, dataset, resp_path, dirty_csv, all_attrs, 
                    related_attrs_dict, {}  # 不使用预函数
                )
            total_time += t.duration
            
            # ==================== 步骤4: 初始化变量 ====================
            labeled_number = 0
            num_epochs = 5000
            
            # 高置信度样本字典
            high_confidence_right_dict = defaultdict(list)
            high_confidence_wrong_dict = defaultdict(list)
            
            # LLM标注历史: {attr: {idx: [label1, label2, ...]}}
            index_value_label_history = defaultdict(lambda: defaultdict(list))
            
            # 训练数据字典: {attr: {'right': [(idx, value)], 'wrong': [(idx, value)]}}
            train_data_dict = defaultdict(lambda: {'right': [], 'wrong': []})
            
            # ==================== 步骤5: 初始LLM多轮标注 ====================
            # 第一次迭代使用聚类中心indices
            indices_dict = {attr: list(clusters[0]) for attr, clusters in cluster_index_dict.items()}
            
            logger.info(f"开始初始LLM多轮标注，共 {INITIAL_LLM_LABEL_ITERATIONS} 轮")
            with Timer('Initial LLM Multi-round Labeling', logger, time_file) as t:
                for round_idx in range(INITIAL_LLM_LABEL_ITERATIONS):
                    logger.info(f"初始标注第 {round_idx + 1} 轮")
                    for attr_name, indices in indices_dict.items():
                        if len(indices) == 0:
                            continue
                        
                        # 调用LLM标注（不使用canonical_patterns作为上下文）
                        # 获取该列的错误模式
                        attr_error_patterns = error_patterns_dict.get(attr_name, [])
                        result = llm_label_indices(
                            attr_name, indices, dirty_csv, clean_csv, related_attrs_dict,
                            high_confidence_right_dict, high_confidence_wrong_dict,
                            error_checking_res_directory, err_check_val_num_per_query,
                            canonical_patterns=None,  # 不作为上下文
                            error_patterns=attr_error_patterns  # 使用错误模式预筛选
                        )
                        
                        # 将结果累积到历史标注中
                        for idx, value, label in result.get(attr_name, []):
                            index_value_label_history[attr_name][idx].append(label)
                    
                    labeled_number += sum(len(indices) for indices in indices_dict.values())
            total_time += t.duration
            
            # 计算并保存每列的总体LLM标注准确率
            logger.info("计算LLM标注总体准确率...")
            for attr in all_attrs:
                error_checking_file = os.path.join(error_checking_res_directory, f'error_checking_{attr}.txt')
                if os.path.exists(error_checking_file):
                    with open(error_checking_file, 'a', encoding='utf-8') as f:
                        f.write("\n" + "="*80 + "\n")
                        f.write(f"OVERALL ACCURACY SUMMARY FOR COLUMN: {attr}\n")
                        f.write("="*80 + "\n")
                        
                        # 从文件中提取所有批次的准确率
                        with open(error_checking_file, 'r', encoding='utf-8') as rf:
                            file_content = rf.read()
                            batch_accuracies = re.findall(r'// Batch Accuracy: (\d+)/(\d+) = ([\d.]+)', file_content)
                            
                            if batch_accuracies:
                                total_correct = sum(int(match[0]) for match in batch_accuracies)
                                total_samples = sum(int(match[1]) for match in batch_accuracies)
                                overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
                                
                                f.write(f"Total Batches: {len(batch_accuracies)}\n")
                                f.write(f"Total Samples: {total_samples}\n")
                                f.write(f"Total Correct: {total_correct}\n")
                                f.write(f"Total Wrong: {total_samples - total_correct}\n")
                                f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n")
                                f.write("="*80 + "\n")
                                
                                logger.info(f"列 '{attr}' LLM标注准确率: {overall_accuracy:.4f}")
            
            # ==================== 步骤6: 根据一致性构建训练集 ====================
            logger.info("根据LLM标注一致性构建训练集")
            with Timer('Building Training Set', logger, time_file) as t:
                train_data_dict, final_labels = convert_label_history_to_train_data(
                    index_value_label_history, dirty_csv, related_attrs_dict,
                    INITIAL_LLM_LABEL_CONSISTENCY_THRESHOLD, all_attrs
                )
                
                # 统计训练数据
                total_train_samples = 0
                for attr in all_attrs:
                    right_count = len(train_data_dict[attr]['right'])
                    wrong_count = len(train_data_dict[attr]['wrong'])
                    total_train_samples += right_count + wrong_count
                    logger.info(f"属性 {attr}: 正确样本 {right_count}, 错误样本 {wrong_count}")
                
                logger.info(f"训练集总样本数: {total_train_samples}")
                para_file.write(f"Training samples: {total_train_samples}\n")
            total_time += t.duration
            
            # ==================== 步骤7: 训练MLP模型 ====================
            logger.info("训练MLP模型")
            model_col = {}
            with Timer('Model Training', logger, time_file) as t:
                feat_dict_train = {}
                label_dict_train = {}
                
                for attr in all_attrs:
                    # 获取该列的标准模式和错误模式（如果有）
                    attr_canonical_patterns = canonical_patterns_dict.get(attr, None)
                    attr_error_patterns = error_patterns_dict.get(attr, [])
                    attr_name, feature_list, label_list = process_attr_train_feat_simplified(
                        attr, dirty_csv, train_data_dict, related_attrs_dict,
                        resp_path, canonical_patterns=attr_canonical_patterns,
                        error_patterns=attr_error_patterns
                    )
                    feat_dict_train[attr] = feature_list
                    label_dict_train[attr] = label_list
                
                for attr in tqdm(all_attrs, desc="Training models", ncols=120):
                    attr_name, model, _, _, _, _ = train_model(
                        attr, feat_dict_train[attr], label_dict_train[attr], num_epochs
                    )
                    if model is not None:
                        model_col[attr] = model
                
                logger.info(f"成功训练 {len(model_col)} 个模型")
            total_time += t.duration
            
            # ==================== 步骤8: 最终预测 ====================
            logger.info("使用模型进行错误检测")
            det_wrong_list_res = []
            with Timer('Final Prediction', logger, time_file) as t:
                for col, attr in tqdm(enumerate(all_attrs), desc="Making predictions", ncols=120):
                    # 获取该列的标准模式和错误模式（如果有）
                    attr_canonical_patterns = canonical_patterns_dict.get(attr, None)
                    attr_error_patterns = error_patterns_dict.get(attr, [])
                    wrong_cells = make_predictions_simplified(
                        col, attr, dirty_csv, model_col, related_attrs_dict,
                        resp_path, canonical_patterns=attr_canonical_patterns,
                        error_patterns=attr_error_patterns
                    )
                    for cell in wrong_cells:
                        if cell not in det_wrong_list_res:
                            det_wrong_list_res.append(cell)
            total_time += t.duration
            
            # ==================== 步骤9: 评估检测结果 ====================
            logger.info("评估错误检测结果")
            det_res_path = os.path.join(resp_path, "detection_results.txt")
            measure_detect(clean_path, dirty_path, list(det_wrong_list_res), det_res_path)
            
            # 打印预测错误详情
            print_prediction_errors(
                dirty_csv, clean_csv, det_wrong_list_res, all_attrs, 
                related_attrs_dict, logger, resp_path
            )
            
            # 保存MLP预测错误
            mlp_errors_file = save_mlp_prediction_errors(
                dirty_csv, clean_csv, det_wrong_list_res, all_attrs, resp_path
            )
            logger.info(f"MLP预测错误已保存到: {mlp_errors_file}")
            
            # ==================== 步骤10: 保存结果 ====================
            logger.info("保存结果文件")
            
            # 保存训练数据
            train_data_save_path = os.path.join(resp_path, 'train_data_dict.json')
            with open(train_data_save_path, 'w', encoding='utf-8') as f:
                serializable_train_data = {}
                for attr, data in train_data_dict.items():
                    serializable_train_data[attr] = {
                        'right': [(int(idx), val) for idx, val in data['right']],
                        'wrong': [(int(idx), val) for idx, val in data['wrong']]
                    }
                json.dump(serializable_train_data, f, ensure_ascii=False, indent=2)
            
            # 保存标注历史
            label_history_path = os.path.join(resp_path, 'label_history.json')
            with open(label_history_path, 'w', encoding='utf-8') as f:
                serializable_history = {}
                for attr, idx_labels in index_value_label_history.items():
                    serializable_history[attr] = {
                        str(idx): labels for idx, labels in idx_labels.items()
                    }
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
            
            # 保存模型
            model_save_path = os.path.join(resp_path, 'models.pkl')
            with open(model_save_path, 'wb') as f:
                pickle.dump(model_col, f)
            
            # ==================== 完成 ====================
            time_end = time.time()
            total_time += time_end - time_start
            
            para_file.write(f"\nTotal LLM labeled samples: {labeled_number}\n")
            para_file.write(f"Training set size: {sum(len(train_data_dict[attr]['right']) + len(train_data_dict[attr]['wrong']) for attr in all_attrs)}\n")
            para_file.write(f"Detected errors: {len(det_wrong_list_res)}\n")
            
            time_file.write(f"total: {total_time:.2f}s\n")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"处理完成!")
            logger.info(f"总耗时: {total_time:.2f}s")
            logger.info(f"LLM标注样本数: {labeled_number}")
            logger.info(f"检测到的错误数: {len(det_wrong_list_res)}")
            logger.info(f"结果保存在: {resp_path}")
            logger.info(f"{'='*60}")
            
            time_file.close()
            para_file.close()

            para_file.close()
