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
                        llm_canonicality_score_prompt
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


def perform_distribution_analysis(dirty_csv, col_num, col_name, config, logger):
    """执行分布分析方法"""
    eps = config.get('eps', 0.3)
    max_cluster_centers = config.get('max_cluster_centers', 20)
    top_canonical_clusters = config.get('top_canonical_clusters', 2)
    alpha = config.get('alpha', 0.4)
    beta = config.get('beta', 0.3)
    gamma = config.get('gamma', 0.3)
    
    logger.info(f"对列 '{col_name}' 执行分布分析，eps={eps}")
    
    cluster_result = single_column_dbscan_clustering(dirty_csv, col_num, col_name, eps=eps)
    
    if cluster_result is None or cluster_result['n_clusters'] == 0:
        logger.warning(f"列 '{col_name}' 聚类结果为空")
        return None
    
    logger.info(f"列 '{col_name}' 聚类完成，共 {cluster_result['n_clusters']} 个聚类")
    
    total_samples = len(dirty_csv)
    canonical_scores = []
    score_components = []
    
    # 从配置中获取delta参数（LLM规范性权重）
    delta = config.get('delta', 0.45)
    
    for cluster_values in cluster_result['cluster_values']:
        score, components = calculate_canonical_score(
            cluster_values, total_samples, alpha, beta, gamma, delta,
            attr_name=col_name, logger=logger, use_llm_score=True
        )
        canonical_scores.append(score)
        score_components.append(components)
    
    canonical_probs = calculate_canonical_probability(canonical_scores)
    
    sorted_indices = np.argsort(canonical_scores)[::-1]
    top_canonical_indices = sorted_indices[:top_canonical_clusters].tolist()
    
    logger.info(f"列 '{col_name}' Top-{top_canonical_clusters} Canonical簇: {top_canonical_indices}")
    for i, idx in enumerate(top_canonical_indices):
        logger.info(f"  Canonical {i+1}: 聚类{idx}, Score={canonical_scores[idx]:.4f}, Prob={canonical_probs[idx]:.4f}")
    
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
        'top_canonical_indices': top_canonical_indices,
        'noise_indices': cluster_result['noise_indices'],
        'config': config
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


def analyze_canonical_patterns_with_llm(analysis_result, dirty_csv, logger):
    """使用LLM分析Canonical簇的标准模式"""
    col_name = analysis_result['col_name']
    top_canonical_indices = analysis_result['top_canonical_indices']
    cluster_values = analysis_result['cluster_values']
    canonical_scores = analysis_result['canonical_scores']
    max_samples = analysis_result['config'].get('max_samples_per_cluster', 10)
    
    canonical_patterns = []
    
    for idx in top_canonical_indices:
        if idx >= len(cluster_values):
            continue
        
        samples = cluster_values[idx][:max_samples]
        score = canonical_scores[idx]
        
        prompt = canonical_pattern_analysis_prompt(col_name, samples, idx, score)
        
        try:
            response = query_base(prompt)
            
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                pattern_json = json_match.group(1)
                pattern = json.loads(pattern_json)
                pattern['cluster_id'] = idx
                pattern['canonical_score'] = score
                canonical_patterns.append(pattern)
                logger.info(f"列 '{col_name}' 聚类{idx} 标准模式: {pattern.get('pattern_name', 'Unknown')}")
            else:
                try:
                    pattern = json.loads(response)
                    pattern['cluster_id'] = idx
                    pattern['canonical_score'] = score
                    canonical_patterns.append(pattern)
                except:
                    logger.warning(f"列 '{col_name}' 聚类{idx} 无法解析LLM响应")
                    canonical_patterns.append({
                        'pattern_name': f'Pattern_{idx}',
                        'pattern_description': f'Cluster {idx} pattern',
                        'regex_pattern': 'N/A',
                        'key_characteristics': [],
                        'example_valid_values': samples[:3],
                        'common_errors': [],
                        'cluster_id': idx,
                        'canonical_score': score
                    })
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
                'canonical_score': score
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


def save_distribution_analysis_results(analysis_results, canonical_patterns_dict, resp_path, logger):
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
    
    return dist_analysis_dir


def process_distribution_analysis_for_all_columns(dirty_csv, all_attrs, config, resp_path, logger):
    """对所有列执行分布分析流程"""
    distribution_analysis_results = {}
    canonical_patterns_dict = {}
    use_distribution_analysis = {}
    
    if not config.get('enabled', False):
        logger.info("分布分析方法未启用")
        for attr in all_attrs:
            use_distribution_analysis[attr] = False
        return distribution_analysis_results, canonical_patterns_dict, use_distribution_analysis
    
    logger.info("开始分布分析流程...")
    
    for col_num, attr in enumerate(all_attrs):
        logger.info(f"\n处理列 '{attr}' ({col_num + 1}/{len(all_attrs)})")
        
        analysis_result = perform_distribution_analysis(dirty_csv, col_num, attr, config, logger)
        
        if analysis_result is None:
            logger.warning(f"列 '{attr}' 分布分析失败，使用原方法")
            use_distribution_analysis[attr] = False
            continue
        
        center_values = analysis_result['center_values'][:config.get('max_cluster_centers', 20)]
        need_analysis = ask_llm_for_distribution_analysis(attr, center_values, logger)
        
        if not need_analysis:
            logger.info(f"列 '{attr}' 不需要分布分析，使用原方法")
            use_distribution_analysis[attr] = False
            distribution_analysis_results[attr] = analysis_result
            continue
        
        logger.info(f"列 '{attr}' 需要分布分析")
        use_distribution_analysis[attr] = True
        distribution_analysis_results[attr] = analysis_result
        
        canonical_patterns = analyze_canonical_patterns_with_llm(analysis_result, dirty_csv, logger)
        canonical_patterns_dict[attr] = canonical_patterns
        
        logger.info(f"列 '{attr}' 分析完成，识别出 {len(canonical_patterns)} 个标准模式")
    
    if distribution_analysis_results:
        save_distribution_analysis_results(
            distribution_analysis_results, canonical_patterns_dict, resp_path, logger
        )
    
    return distribution_analysis_results, canonical_patterns_dict, use_distribution_analysis


# ==================== 分布分析方法相关函数结束 ====================



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


def calculate_prediction_stability(pred_history):
    """
    计算预测稳定性 Stab(x) = 1 - Var(p(1), ..., p(T))
    
    Args:
        pred_history: 预测概率历史列表
    
    Returns:
        stability: 稳定性分数
    """
    if not pred_history or len(pred_history) < 2:
        return 1.0
    
    variance = np.var(pred_history)
    stability = 1 - min(variance, 1.0)  # 确保稳定性在0-1之间
    return stability


def calculate_confidence_score(p, y_mlp_equals_y_llm, llm_consistency, alpha=0.7, beta=1.2):
    """
    计算置信度分数
    Conf(x) = σ(α|p-0.5|(yMLP==yLLM) + β * CLLM-cons(x))
    
    Args:
        p: 模型预测概率
        y_mlp_equals_y_llm: MLP和LLM标注是否一致 (1或0)
        llm_consistency: LLM标注一致性分数
        alpha: α参数
        beta: β参数
    
    Returns:
        confidence: 置信度分数
    """
    score = alpha * abs(p - 0.5) * y_mlp_equals_y_llm + beta * llm_consistency
    confidence = sigmoid(score)
    return confidence


def calculate_training_confidence(p, stability, alpha=0.7):
    """
    计算训练数据置信度
    Conf(x) = α * 2 * |p - 0.5| + (1 - α) * Stab(x)
    
    Args:
        p: 模型预测概率 (范围 [0, 1])
        stability: 预测稳定性 (范围 [0, 1])
        alpha: α参数，控制预测置信度和稳定性的权重
    
    Returns:
        confidence: 置信度分数 (范围 [0, 1])
    """
    # 2 * |p - 0.5| 将 [0, 0.5] 归一化到 [0, 1]
    normalized_pred_confidence = 2 * abs(p - 0.5)
    confidence = alpha * normalized_pred_confidence + (1 - alpha) * stability
    return confidence


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


def filter_unlabeled_indices(indices_dict, index_value_label_history):
    """
    过滤掉已经标注过的索引
    
    Args:
        indices_dict: {attr: [idx1, idx2, ...]}
        index_value_label_history: {attr: {idx: [label1, label2, ...]}}
    
    Returns:
        filtered_indices_dict: 过滤后的索引字典
        labeled_count: 已标注的数量
    """
    filtered_indices_dict = {}
    labeled_count = 0
    
    for attr, indices in indices_dict.items():
        labeled_indices = set(index_value_label_history.get(attr, {}).keys())
        unlabeled_indices = [idx for idx in indices if idx not in labeled_indices]
        filtered_indices_dict[attr] = unlabeled_indices
        labeled_count += len(indices) - len(unlabeled_indices)
    
    return filtered_indices_dict, labeled_count


def get_indices_from_optimal_cluster(optimal_cluster_info_dict):
    """
    从最优聚类信息中提取索引
    
    Args:
        optimal_cluster_info_dict: {attr: {'cluster_idx': ..., 'cluster_indices': [...], ...}}
    
    Returns:
        indices_dict: {attr: [idx1, idx2, ...]}
    """
    indices_dict = {}
    for attr, cluster_info in optimal_cluster_info_dict.items():
        if cluster_info is not None:
            indices_dict[attr] = list(cluster_info['cluster_indices'])
        else:
            indices_dict[attr] = []
    return indices_dict


def make_predictions_with_proba(col, attr, dirty_csv, model_col, related_attrs_dict, 
                                 funcs_for_attr, feature_all_dict, resp_path, canonical_patterns=None):
    """
    预测并返回概率值
    
    Returns:
        predictions: [(idx, attr, pred_label, pred_proba), ...]
    """
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
    pred_labels = model.predict(test_feat_np)
    
    # 获取预测概率
    if hasattr(model, 'predict_proba'):
        pred_probas = model.predict_proba(test_feat_np)
        # 取正类（错误类）的概率
        pred_probas = pred_probas[:, 1] if pred_probas.shape[1] > 1 else pred_probas[:, 0]
    else:
        pred_probas = pred_labels.astype(float)
    
    predictions = []
    for idx in range(len(dirty_csv)):
        predictions.append((idx, attr, pred_labels[idx], pred_probas[idx]))
    
    return predictions


def save_confidence_samples(high_conf_dict, mid_conf_dict, save_path):
    """
    保存高置信度和中等置信度样本到文件
    """
    with open(os.path.join(save_path, 'high_confidence_samples.json'), 'w', encoding='utf-8') as f:
        # 转换为可序列化格式
        serializable = {}
        for attr, samples in high_conf_dict.items():
            serializable[attr] = {
                'right': [(int(idx), val) for idx, val in samples.get('right', [])],
                'wrong': [(int(idx), val) for idx, val in samples.get('wrong', [])]
            }
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(save_path, 'mid_confidence_samples.json'), 'w', encoding='utf-8') as f:
        serializable = {}
        for attr, samples in mid_conf_dict.items():
            serializable[attr] = {
                'right': [(int(idx), val) for idx, val in samples.get('right', [])],
                'wrong': [(int(idx), val) for idx, val in samples.get('wrong', [])]
            }
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def compute_f1_score_for_iteration(model_col, dirty_csv, clean_csv, all_attrs, related_attrs_dict, 
                                    funcs_for_attr, feature_all_dict, resp_path, logger, canonical_patterns_dict=None):
    """
    使用当前模型对整个脏数据进行预测并计算F1分数
    
    Returns:
        precision, recall, f1_score, detected_errors, total_errors
    """
    det_wrong_list = []
    
    # 对每个属性进行预测
    for col, attr in enumerate(all_attrs):
        if attr not in model_col:
            continue
        
        # 获取该列的标准模式（如果有）
        attr_canonical_patterns = canonical_patterns_dict.get(attr, None) if canonical_patterns_dict else None
        wrong_cells = make_predictions(
            col, attr, dirty_csv, model_col, related_attrs_dict,
            funcs_for_attr, feature_all_dict, resp_path,
            canonical_patterns=attr_canonical_patterns
        )
        det_wrong_list.extend(wrong_cells)
    
    # 计算真实错误数
    total_errors = 0
    for attr in all_attrs:
        for idx in range(len(dirty_csv)):
            if str(dirty_csv.loc[idx, attr]) != str(clean_csv.loc[idx, attr]):
                total_errors += 1
    
    # 计算真阳性（正确检测到的错误）
    true_positives = 0
    for idx, attr in det_wrong_list:
        if str(dirty_csv.loc[idx, attr]) != str(clean_csv.loc[idx, attr]):
            true_positives += 1
    
    detected_errors = len(det_wrong_list)
    
    # 计算精确率、召回率和F1分数
    precision = true_positives / detected_errors if detected_errors > 0 else 0
    recall = true_positives / total_errors if total_errors > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"  检测到的错误数: {detected_errors}, 真实错误数: {total_errors}")
    logger.info(f"  精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1_score:.4f}")
    
    return precision, recall, f1_score, detected_errors, total_errors


# ==================== 原有函数（部分修改） ====================

def llm_label_indices(attr_name, indices, dirty_csv, related_attrs_dict, 
                      high_confidence_right_dict, high_confidence_wrong_dict,
                      error_checking_res_directory, err_check_val_num_per_query=20,
                      canonical_patterns=None):
    """
    对指定的indices进行LLM标注，累积保存标注文件，并返回当前标注结果
    
    Returns:
        current_labels: {attr: [(idx, value, label), ...]}
    """
    related_attrs = list(related_attrs_dict[attr_name])
    
    # 为每个索引创建数据字典
    df_indices = ["{" + ",".join(f'"{col}":"{dirty_csv.loc[idx, col]}"' for col in [attr_name] + related_attrs) + "}" for idx in indices]
    
    # 将数据分成子列表进行处理
    split_values = split_list_to_sublists(df_indices, err_check_val_num_per_query)
    split_indices = split_list_to_sublists(list(indices), err_check_val_num_per_query)
    
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
                f.write(response + '\n\n')
            
            all_responses.append((response, sub_list_indices))
            
        except Exception as e:
            print(f"处理属性 {attr_name} 的子任务时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    current_labels = extract_labels_from_responses(attr_name, all_responses, dirty_csv, related_attrs_dict)
    
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


def gen_dirty_funcs(attr, clean_info, errs_info, api_use, model_type):
    dirty_str = "\n"
    
    # 确保clean_info是可迭代的列表
    if clean_info is None:
        clean_info = []
    if not isinstance(clean_info, (list, tuple)):
        clean_info = [clean_info]
    clean_info_str = '\n'.join([str(i) for i in clean_info if i is not None])
    
    # 确保errs_info是可迭代的列表
    if errs_info is None:
        errs_info = []
    if not isinstance(errs_info, (list, tuple)):
        errs_info = [errs_info]
    
    try:
        dirty_str = dirty_str + '\n'.join([str(i) for i in errs_info if i is not None])
    except Exception as e:
        print(f"Error: {e}\n When handling {errs_info}\n")
        dirty_str = dirty_str + str(errs_info) + "\n"
    
    func_gen_prompt = err_clean_func_prompt(attr, clean_info_str, dirty_str)
    llm_gen_func = get_ans_from_llm(func_gen_prompt, api_use=api_use, model_type=model_type)
    
    # 确保llm_gen_func是字符串
    if llm_gen_func is None:
        llm_gen_func = ""
    
    temp_clean_flist, dirty_flist = extract_func(llm_gen_func)
    return temp_clean_flist, dirty_flist, func_gen_prompt, llm_gen_func


def subtask_func_gen(attr_name, err_list, func_file_num, right_values_list, funcs_directory, api_use, model_type):
    temp_clean_flist, dirty_flist, func_gen_prompt, llm_gen_func = gen_dirty_funcs(attr_name, right_values_list, err_list, api_use, model_type)
    funcs_for_attr = defaultdict(default_dict_of_lists)
    funcs_for_attr[attr_name]['clean'].extend(list(set(temp_clean_flist)))
    funcs_for_attr[attr_name]['dirty'].extend(list(set(dirty_flist)))
    with open(os.path.join(funcs_directory, f"prompt_funcs_zgen_{attr_name}{func_file_num}.txt"), 'w', encoding='utf-8') as prom_file:
        prom_file.write(func_gen_prompt)
    with open(os.path.join(funcs_directory, f"funcs_zgen_{attr_name}{func_file_num}.txt"), 'w', encoding='utf-8') as func_file:
        func_file.write("\n".join(list(set(temp_clean_flist))))
    return attr_name, funcs_for_attr


def gen_err_funcs(attr, high_confidence_right_dict, high_confidence_wrong_dict, dirty_csv, related_attrs_dict, funcs_directory, api_use, model_type):
    """根据高置信度样本生成错误检测函数"""
    related_attrs = list(related_attrs_dict[attr])
    
    wrong_values = []
    right_values = []
    
    # 从高置信度字典获取数据，确保数据有效
    if attr in high_confidence_wrong_dict:
        wrong_values = [v for v in high_confidence_wrong_dict[attr] if v is not None]
    if attr in high_confidence_right_dict:
        right_values = [v for v in high_confidence_right_dict[attr] if v is not None]
    
    # 将值转换为字符串，处理可能的异常
    filtered_error = []
    for vals in wrong_values:
        try:
            filtered_error.append(str(vals))
        except Exception as e:
            print(f"Warning: Cannot convert value to string: {e}")
            continue
    
    if len(filtered_error) == 0:
        return False
    
    max_err_num = 20
    if max_err_num > (int(len(filtered_error)/2)+1):
        max_err_num = int(len(filtered_error)/2)+1
    filtered_error_sublists = split_list_to_sublists(filtered_error, max_err_num)
    if len(filtered_error_sublists) > 2:
        filtered_error_sublists = filtered_error_sublists[:2]
    
    funcs_for_attr = {}
    max_err_num = min(max_err_num, len(right_values)) if right_values else 1
    
    with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as a_executor:
        a_results = []
        for temp_idx in range(len(filtered_error_sublists)):
            sample_right = random.sample(right_values, min(max_err_num, len(right_values))) if right_values else []
            a_results.append(a_executor.submit(
                subtask_func_gen, attr, filtered_error_sublists[temp_idx], 
                temp_idx, sample_right, funcs_directory, api_use, model_type
            ))
        for a_future in as_completed(a_results):
            attr_name, funcs_for_attr_gen = a_future.result()
            funcs_for_attr.update(funcs_for_attr_gen)
    
    func_extract_file = open(os.path.join(funcs_directory, f"funcs_zgen_{attr}.txt"), 'w', encoding='utf-8')
    if attr in funcs_for_attr:
        temp_clean_flist_str = "\n".join(funcs_for_attr[attr]['clean'])
        func_extract_file.write(temp_clean_flist_str)
    func_extract_file.close()
    return funcs_for_attr


def execute_func(function_code, val, attr):
    local_scope = {}
    exec(function_code, globals(), local_scope)
    function_name = list(local_scope.keys())[0]
    function = local_scope[function_name]
    return function(val, attr)


funcs_with_errors = set()

def handle_func_exec(func, val, attr):
    try:
        result = execute_func(func, val, attr)
    except Exception as err:
        func_str = f"Error: {err}\n" + f"Value: {val}, Attribute: {attr}\nFunc: {func}\n"
        funcs_with_errors.add(func_str)
        return -1
    return 1 if result else 0


def task_func_gen(attr_name, high_confidence_right_dict, high_confidence_wrong_dict, dirty_csv, related_attrs_dict, 
                  funcs_directory, para_file, api_use, model_type):
    funcs_for_attr = gen_err_funcs(attr_name, high_confidence_right_dict, high_confidence_wrong_dict, dirty_csv, related_attrs_dict, 
                                    funcs_directory, api_use, model_type)
    if funcs_for_attr:
        para_file.write(f"{attr_name} func_num:{len(funcs_for_attr.get(attr_name, {}).get('clean', []))}\n")
        return funcs_for_attr
    else:
        return {attr_name: {'clean': [], 'dirty': []}}


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


def process_select_optimal_cluster(
    train_data_dict, cluster_index_dict, dirty_csv, all_attrs, related_attrs_dict,
    pre_funcs_for_attr, resp_path, logger, residual_method='both',
    previously_selected_clusters=None, cluster_selection_window=-1
):
    """
    从聚类结果中选出最优聚类，返回indices_dict格式
    
    Args:
        train_data_dict: 训练数据字典 {attr: {'right': [...], 'wrong': [...]}}
        cluster_index_dict: 聚类索引字典
        dirty_csv: 脏数据DataFrame
        all_attrs: 所有属性列表
        related_attrs_dict: 相关属性字典
        pre_funcs_for_attr: 预处理函数字典
        resp_path: 响应路径
        logger: 日志记录器
        index_value_label_history: 历史标注 {attr: {idx: [labels]}}
        residual_method: 残差计算方法
        previously_selected_clusters: 之前选择过的聚类 {attr: [cluster_idx1, cluster_idx2, ...]}
        cluster_selection_window: 聚类选择窗口大小
            -1: 不能选择任何选择过的聚类（默认行为）
            1: 不能选择上一次选择的聚类
            2: 不能选择前两次选择的聚类
            以此类推
    
    Returns:
        indices_dict: {attr: [idx1, idx2, ...]} 最优聚类的索引
    """
    optimal_cluster_info_dict = {}
    
    if previously_selected_clusters is None:
        previously_selected_clusters = {}
    
    logger.info(f"开始选择最优聚类... (窗口大小: {cluster_selection_window}")
    
    try:
        fasttext_model = fasttext.load_model('./cc.en.300.bin')
        fasttext_dimension = len(related_attrs_dict[next(iter(related_attrs_dict))]) + 1
        fasttext.util.reduce_model(fasttext_model, fasttext_dimension)
    except Exception as e:
        logger.error(f"加载FastText模型失败: {str(e)}")
        return {}

    global_cache = feat_gen_global_cache(dirty_csv, related_attrs_dict)

    def process_attr(attr):
        min_residual = float('inf')
        attr_optimal_cluster = None
        
        logger.info(f"处理属性: {attr}")
        
        if attr not in cluster_index_dict:
            logger.warning(f"属性 {attr} 不在聚类索引字典中，跳过")
            return None
            
        clusters = cluster_index_dict[attr]
        if len(clusters) == 0:
            logger.warning(f"属性 {attr} 没有有效聚类，跳过")
            return None
            
        related_attrs = list(related_attrs_dict[attr])
        col_num = list(dirty_csv.columns).index(attr)

        ref_data = dirty_csv.loc[:, [attr] + related_attrs]
        ref_df = pd.DataFrame(ref_data) if not ref_data.empty else pd.DataFrame()
        col_num = list(ref_df.columns).index(attr)
        ref_features, _, scaler = feat_gen_df_incremental(ref_df, col_num, attr, pre_funcs_for_attr, resp_path, global_cache)
        ref_features = np.array(ref_features, dtype=np.float64)
        ref_features = np.nan_to_num(ref_features)

        attr_previously_selected = previously_selected_clusters.get(attr, [])
        
        # 根据窗口大小确定需要排除的聚类
        excluded_clusters = set()
        if cluster_selection_window == -1:
            # -1 表示不能选择任何选择过的聚类
            excluded_clusters = set(attr_previously_selected)
        elif cluster_selection_window > 0:
            # 只排除最近 window_size 次选择的聚类
            excluded_clusters = set(attr_previously_selected[-cluster_selection_window:])
        # 如果 window_size == 0，不排除任何聚类
        
        for cluster_idx, cluster_indices in enumerate(clusters[1:], start=0):
            if cluster_idx in excluded_clusters:
                logger.info(f"跳过属性 {attr} 的聚类 {cluster_idx}，因为在窗口范围内已经选择过")
                continue
                
            if len(cluster_indices) == 0:
                continue
                
            cluster_data = dirty_csv.loc[cluster_indices, [attr] + related_attrs]
            
            # 合并训练数据
            train_data = []
            if attr in train_data_dict:
                for idx, val in train_data_dict[attr].get('right', []):
                    if idx not in cluster_indices:
                        train_data.append(val)
                for idx, val in train_data_dict[attr].get('wrong', []):
                    if idx not in cluster_indices:
                        train_data.append(val)
            train_df = pd.DataFrame(train_data) if train_data else pd.DataFrame()
            
            combined_data = pd.concat([cluster_data, train_df], ignore_index=True)
            if combined_data.empty:
                continue

            combined_feature_list, _, _ = feat_gen_df_incremental(
                combined_data, col_num, attr, pre_funcs_for_attr, resp_path, global_cache, scaler
            )
            combined_feature_list = np.nan_to_num(combined_feature_list)
            
            try:
                if residual_method in ['jsd', 'both']:
                    hist_comb, _ = np.histogram(combined_feature_list.flatten(), bins=30, density=True)
                    hist_ref, _ = np.histogram(ref_features.flatten(), bins=30, density=True)
                    jsd_residual = calculate_jsd(hist_comb, hist_ref)
                else:
                    jsd_residual = float('inf')

                if residual_method in ['ksd', 'both']:
                    if combined_feature_list.ndim > 1:
                        mean_comb = np.mean(combined_feature_list, axis=1)
                    else:
                        mean_comb = combined_feature_list
                    
                    if ref_features.ndim > 1:
                        mean_ref = np.mean(ref_features, axis=1)
                    else:
                        mean_ref = ref_features
                    
                    ksd_residual = calculate_ksd(mean_comb, mean_ref)
                else:
                    ksd_residual = float('inf')

                if residual_method == 'both':
                    combined_residual = 0.5 * jsd_residual + 0.5 * ksd_residual
                elif residual_method == 'jsd':
                    combined_residual = jsd_residual
                else:
                    combined_residual = ksd_residual

                logger.info(f"{attr} 聚类 {cluster_idx}: JSD={jsd_residual:.4f}, KSD={ksd_residual:.4f}, 综合={combined_residual:.4f}")

                if combined_residual < min_residual:
                    min_residual = combined_residual
                    attr_optimal_cluster = {
                        'cluster_idx': cluster_idx,
                        'cluster_indices': cluster_indices,
                        'jsd_residual': jsd_residual,
                        'ksd_residual': ksd_residual,
                        'combined_residual': combined_residual
                    }

            except Exception as e:
                logger.error(f"计算属性 {attr} 聚类 {cluster_idx} 残差时出错: {str(e)}")
                continue

        return attr, attr_optimal_cluster

    for attr in all_attrs:
        result = process_attr(attr)
        if result:
            attr, attr_optimal_cluster = result
            if attr_optimal_cluster is not None:
                optimal_cluster_info_dict[attr] = attr_optimal_cluster

    # 转换为indices_dict格式
    indices_dict = get_indices_from_optimal_cluster(optimal_cluster_info_dict)
    
    if optimal_cluster_info_dict:
        logger.info("最优聚类信息:")
        for attr, cluster_info in optimal_cluster_info_dict.items():
            logger.info(f"属性 {attr} 聚类 {cluster_info['cluster_idx']}，综合残差 {cluster_info['combined_residual']:.4f}")
    else:
        logger.warning("未找到有效的最优聚类")
    
    return indices_dict


def measure_llm_label(resp_path, clean_csv, all_attrs, related_attrs_dict, gt_wrong_dict, final_labels):
    """
    评估LLM标注结果
    
    Args:
        final_labels: {attr: [(idx, value, label), ...]} 最终标签
    """
    llm_label_eval_file = open(os.path.join(resp_path, 'llm_label_results.txt'), 'w', encoding='utf-8')
    overall_wrong_label_num = 0
    overall_lwrong_num = 0
    overall_lright_num = 0
    overall_miss_wrong_num = 0
    
    for attr in all_attrs:
        llm_label_eval_file.write('\n' + '*'*30 + attr + '*'*30 + '\n\n')
        wrongly_llm_det = []
        missing_llm_det = []
        llm_wrong_label_num = 0
        llm_lwrong_num = 0
        llm_lright_num = 0
        llm_miss_wrong_num = 0
        
        for idx, llm_lstr, llm_label in final_labels.get(attr, []):
            if llm_label == 1:
                llm_lwrong_num += 1
                overall_lwrong_num += 1
                if str(llm_lstr) not in gt_wrong_dict[attr]:
                    llm_wrong_label_num += 1
                    overall_wrong_label_num += 1
                    wrongly_llm_det.append((idx, str(llm_lstr)))
            elif llm_label == 0:
                llm_lright_num += 1
                overall_lright_num += 1
                if str(llm_lstr) in gt_wrong_dict[attr]:
                    llm_miss_wrong_num += 1
                    overall_miss_wrong_num += 1
                    missing_llm_det.append((idx, str(llm_lstr)))
        
        llm_label_eval_file.write(f"Wrong data labeling accuracy: {1-llm_wrong_label_num/(llm_lwrong_num+1e-6)} ({llm_lwrong_num-llm_wrong_label_num}/{llm_lwrong_num})\n")
        llm_label_eval_file.write(f"Right data labeling accuracy: {1-llm_miss_wrong_num/(llm_lright_num+1e-6)} ({llm_lright_num-llm_miss_wrong_num}/{llm_lright_num})\n\n")
        llm_label_eval_file.write('-'*30 + "Wrongly Detected Values" + '-'*30 + '\n\n')
        for idx, llm_lstr in wrongly_llm_det:
            llm_label_eval_file.write('\nDirty: ' + llm_lstr)
            llm_label_eval_file.write('\nClean: ' + str(clean_csv.loc[int(idx), [attr] + list(related_attrs_dict[attr])].to_dict()) + '\n')
                
        llm_label_eval_file.write('\n' + '-'*30 + "Missing Erroneous Values" + '-'*30 + '\n\n')
        for idx, llm_lstr in missing_llm_det:
            llm_label_eval_file.write('\nDirty: ' + llm_lstr)
            llm_label_eval_file.write('\nClean: ' + str(clean_csv.loc[int(idx), [attr] + list(related_attrs_dict[attr])].to_dict()) + '\n\n')

    llm_label_eval_file.write('*'*30 + "Overall Evaluation" + '*'*30 + '\n\n')
    llm_label_eval_file.write(f"Overall Wrong data labeling accuracy: {1-overall_wrong_label_num/(overall_lwrong_num+1e-6)} ({overall_lwrong_num-overall_wrong_label_num}/{overall_lwrong_num})\n")
    llm_label_eval_file.write(f"Overall Right data labeling accuracy: {1-overall_miss_wrong_num/(overall_lright_num+1e-6)} ({overall_lright_num-overall_miss_wrong_num}/{overall_lright_num})\n\n")
    llm_label_eval_file.close()
    return 'Done'


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


def compare_llm_and_classifier_labels(index_value_label_history, det_wrong_list_res, dirty_csv, related_attrs_dict):
    """
    比较LLM标注结果和分类器标注结果
    
    Args:
        index_value_label_history: {attr: {idx: [label1, label2, ...]}}
        det_wrong_list_res: [(idx, attr), ...]
    
    Returns:
        comparison_results: {attr: {idx: {'llm_label': label, 'mlp_label': label, 'consistent': bool}}}
    """
    comparison_results = defaultdict(dict)
    
    classifier_labels = {}
    for idx, attr in det_wrong_list_res:
        classifier_labels[(idx, attr)] = 1
    
    for attr, idx_labels in index_value_label_history.items():
        related_attrs = list(related_attrs_dict[attr])
        for idx, label_list in idx_labels.items():
            if not label_list:
                continue
            
            _, llm_majority_label = calculate_llm_consistency(label_list)
            mlp_label = classifier_labels.get((idx, attr), 0)
            
            comparison_results[attr][idx] = {
                'llm_label': llm_majority_label,
                'mlp_label': mlp_label,
                'consistent': llm_majority_label == mlp_label,
                'value': dirty_csv.loc[idx, [attr] + related_attrs].to_dict()
            }
    
    return comparison_results


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


def label_prop(resp_path, dirty_path, clean_path, cluster_index_dict, final_labels, label_prop_flag=True):
    """根据标注结果在聚类内扩散标签"""
    det_wrong_list = []
    det_right_list = []
    
    for attr, label_list in final_labels.items():
        for idx, value, label in label_list:
            if label == 1:
                det_wrong_list.append((idx, attr))
            elif label == 0:
                det_right_list.append((idx, attr))
    
    if not label_prop_flag:
        return det_wrong_list, det_right_list
    
    for attr, clusters in cluster_index_dict.items():
        center_indices = clusters[0]
        
        center_labels = {}
        for idx, value, label in final_labels.get(attr, []):
            if idx in center_indices:
                center_labels[idx] = label
        
        for center_idx, center_label in center_labels.items():
            target_cluster = None
            for i in range(1, len(clusters)):
                if center_idx in clusters[i]:
                    target_cluster = clusters[i]
                    break
            
            if target_cluster is not None:
                labeled_indices = {idx for idx, _, _ in final_labels.get(attr, [])}
                for idx in target_cluster:
                    if idx not in labeled_indices:
                        if center_label == 1:
                            det_wrong_list.append((idx, attr))
                        else:
                            det_right_list.append((idx, attr))
    
    return det_wrong_list, det_right_list


def process_gen_err_funcs(FUNC_USE, resp_path, funcs_directory, dirty_csv, all_attrs, 
                          para_file, related_attrs_dict, high_confidence_right_dict, high_confidence_wrong_dict, api_use, model_type):
    """生成错误检测函数"""
    err_gen_dict = defaultdict(default_dict_of_lists)
    funcs_for_attr = defaultdict(default_dict_of_lists)
    
    if FUNC_USE:
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(task_func_gen, attr, high_confidence_right_dict, high_confidence_wrong_dict, dirty_csv, 
                                       related_attrs_dict, funcs_directory, para_file, 
                                       api_use, model_type) for attr in all_attrs]
            outputs = [result.result() for result in results]
            for output in outputs:
                funcs_for_attr.update(output)
    
    return err_gen_dict, funcs_for_attr


def process_gen_clean_funcs(PRE_FUNC_USE, funcs_pre_directory, dirty_csv, all_attrs, 
                            related_attrs_dict, logger, api_use, model_type):
    """生成预处理函数"""
    pre_funcs_for_attr = defaultdict(default_dict_of_lists)
    
    if PRE_FUNC_USE:
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(gen_clean_funcs, attr, dirty_csv, funcs_pre_directory, 
                                       related_attrs_dict, logger, api_use, model_type) for attr in all_attrs]
            outputs = [result.result() for result in results]
            for output in outputs:
                pre_funcs_for_attr.update(output)
    else:
        for attr in all_attrs:
            pre_funcs_for_attr[attr] = {'clean': []}
    
    return pre_funcs_for_attr


def gen_clean_funcs(attr, dirty_csv, funcs_pre_directory, related_attrs_dict, logger, api_use, model_type):
    """生成清洁函数"""
    related_attrs = list(related_attrs_dict[attr])
    sample_rows = []
    total_rows = len(dirty_csv)
    max_samp_num = 20
    
    if total_rows > 0:
        sample_indices = random.sample(range(total_rows), min(max_samp_num, total_rows))
        for idx in sample_indices:
            row_dict = dirty_csv.loc[idx, [attr] + related_attrs].to_dict()
            sample_rows.append(row_dict)
    
    sample_rows_str = '\n'.join([str(row) for row in sample_rows])
    
    if len(sample_rows) == 0:
        logger.error("The Data is EMPTY!!!")
        return {attr: {'clean': []}}
    
    prompt = pre_func_prompt(attr, sample_rows_str)
    pre_func_response = get_ans_from_llm(prompt, api_use=api_use, model_type=model_type)
    flist, _ = extract_func(pre_func_response)
    
    with open(os.path.join(funcs_pre_directory, f"prompt_pre_funcs_zgen_{attr}.txt"), 'w', encoding='utf-8') as prom_file:
        prom_file.write(prompt)
    with open(os.path.join(funcs_pre_directory, f"pre_funcs_zgen_{attr}.txt"), 'w', encoding='utf-8') as func_file:
        func_file.write("\n".join(list(set(flist))))
    
    return {attr: {'clean': flist}}


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
            
            # ==================== 步骤1: 计算相关属性 ====================
            related_attrs_dict, gt_wrong_dict = {}, {}
            with Timer('Getting Related Attributes', logger, time_file) as t:
                related_attrs_dict, gt_wrong_dict = process_related_attr(
                    RELATED_ATTRS, REL_TOP, resp_path, clean_csv, dirty_csv, all_attrs
                )
            total_time += t.duration

            # ==================== 步骤2: 预函数生成 ====================
            pre_funcs_for_attr = {}
            with Timer('Preliminary Function Generation', logger, time_file) as t:
                pre_funcs_for_attr = process_gen_clean_funcs(
                    PRE_FUNC_USE, funcs_pre_directory, dirty_csv, all_attrs, 
                    related_attrs_dict, logger, API_USE, MODEL_TYPE
                )
            total_time += t.duration
            
            
            # ==================== 步骤2.5: 分布分析（可选） ====================
            distribution_analysis_results = {}
            canonical_patterns_dict = {}
            use_distribution_analysis = {}
            
            if DISTRIBUTION_ANALYSIS_CONFIG.get('enabled', False):
                logger.info("分布分析方法已启用")
                with Timer('Distribution Analysis', logger, time_file) as t:
                    distribution_analysis_results, canonical_patterns_dict, use_distribution_analysis = \
                        process_distribution_analysis_for_all_columns(
                            dirty_csv, all_attrs, DISTRIBUTION_ANALYSIS_CONFIG, resp_path, logger
                        )
                total_time += t.duration
                
                # 记录哪些列使用了分布分析
                dist_analysis_attrs = [attr for attr, use in use_distribution_analysis.items() if use]
                logger.info(f"使用分布分析的列: {dist_analysis_attrs}")
                para_file.write(f"Distribution analysis enabled for: {dist_analysis_attrs}\n")
            else:
                logger.info("分布分析方法未启用，使用原方法")
                for attr in all_attrs:
                    use_distribution_analysis[attr] = False
            
            # ==================== 步骤3: 聚类 ====================
            cluster_index_dict, center_value_dict = {}, {}
            feature_all_dict = defaultdict(default_dict_of_lists)
            with Timer('Clustering', logger, time_file) as t:
                cluster_index_dict, center_value_dict, feature_all_dict = process_cluster(
                    CLUSTER_RATE, dataset, resp_path, dirty_csv, all_attrs, 
                    related_attrs_dict, pre_funcs_for_attr
                )
            total_time += t.duration
            
            # ==================== 步骤4: 初始化变量 ====================
            labeled_number = 0
            num_epochs = 5000
            
            # 高置信度和中等置信度样本字典
            high_confidence_right_dict = defaultdict(list)
            high_confidence_wrong_dict = defaultdict(list)
            mid_confidence_right_dict = defaultdict(list)
            mid_confidence_wrong_dict = defaultdict(list)
            
            # LLM标注历史: {attr: {idx: [label1, label2, ...]}}
            index_value_label_history = defaultdict(lambda: defaultdict(list))
            
            # 模型预测历史: {attr: {idx: [proba1, proba2, ...]}}
            model_prediction_history = defaultdict(lambda: defaultdict(list))
            
            # 训练数据字典: {attr: {'right': [(idx, value)], 'wrong': [(idx, value)]}}
            train_data_dict = defaultdict(lambda: {'right': [], 'wrong': []})
            
            # 已选择的聚类记录
            previously_selected_clusters = defaultdict(list)
            
            # 函数字典
            funcs_for_attr = defaultdict(default_dict_of_lists)
            
            # 模型字典
            model_col = {}
            
            # F1分数记录列表（如果启用每轮计算F1）
            f1_scores_per_iteration = []
            
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
                        
                        # 调用LLM标注
                        # 获取该列的标准模式（如果有）
                        attr_canonical_patterns = canonical_patterns_dict.get(attr_name, None)
                        
                        result = llm_label_indices(
                            attr_name, indices, dirty_csv, related_attrs_dict,
                            high_confidence_right_dict, high_confidence_wrong_dict,
                            error_checking_res_directory, err_check_val_num_per_query,
                            canonical_patterns=attr_canonical_patterns
                        )
                        
                        # 将结果累积到历史标注中
                        for idx, value, label in result.get(attr_name, []):
                            index_value_label_history[attr_name][idx].append(label)
                    
                    labeled_number += sum(len(indices) for indices in indices_dict.values())
            total_time += t.duration
            
            # ==================== 步骤6: 根据一致性构建初始训练集 ====================
            logger.info("根据LLM标注一致性构建初始训练集")
            with Timer('Building Initial Training Set', logger, time_file) as t:
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
                
                logger.info(f"初始训练集总样本数: {total_train_samples}")
                para_file.write(f"Initial training samples: {total_train_samples}\n")
                
                # 如果开启了result_analyze，保存初始训练集的详细信息
                if RESULT_ANALYZE:
                    train_change_dir = os.path.join(resp_path, 'train_set_changes')
                    os.makedirs(train_change_dir, exist_ok=True)
                    
                    initial_train_data = {'phase': 'initial', 'data': {}}
                    for attr in all_attrs:
                        related_attrs = list(related_attrs_dict[attr])
                        initial_train_data['data'][attr] = {'right': [], 'wrong': []}
                        
                        # 保存right样本及其LLM一致性分数
                        for idx, value in train_data_dict[attr]['right']:
                            label_history = index_value_label_history[attr].get(idx, [])
                            consistency, majority_label = calculate_llm_consistency(label_history)
                            majority_count = sum(1 for l in label_history if l == majority_label)
                            total_count = len(label_history)
                            initial_train_data['data'][attr]['right'].append({
                                'idx': int(idx),
                                'value': value,
                                'llm_consistency': consistency,
                                'consistency_formula': f"{majority_count}/{total_count}",
                                'label_history': label_history
                            })
                        
                        # 保存wrong样本及其LLM一致性分数
                        for idx, value in train_data_dict[attr]['wrong']:
                            label_history = index_value_label_history[attr].get(idx, [])
                            consistency, majority_label = calculate_llm_consistency(label_history)
                            majority_count = sum(1 for l in label_history if l == majority_label)
                            total_count = len(label_history)
                            initial_train_data['data'][attr]['wrong'].append({
                                'idx': int(idx),
                                'value': value,
                                'llm_consistency': consistency,
                                'consistency_formula': f"{majority_count}/{total_count}",
                                'label_history': label_history
                            })
                    
                    # 保存初始训练集
                    initial_file = os.path.join(train_change_dir, 'initial_train_set.json')
                    with open(initial_file, 'w', encoding='utf-8') as f:
                        json.dump(initial_train_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"初始训练集详细信息已保存到: {initial_file}")
            total_time += t.duration
            
            # ==================== 步骤7: 初始模型训练 ====================
            logger.info("训练初始模型")
            with Timer('Initial Model Training', logger, time_file) as t:
                feat_dict_train = {}
                label_dict_train = {}
                
                for attr in all_attrs:
                    # 获取该列的标准模式（如果有）
                    attr_canonical_patterns = canonical_patterns_dict.get(attr, None)
                    attr_name, feature_list, label_list = process_attr_train_feat(
                        attr, dirty_csv, train_data_dict, related_attrs_dict,
                        funcs_for_attr, feature_all_dict, resp_path,
                        canonical_patterns=attr_canonical_patterns
                    )
                    feat_dict_train[attr] = feature_list
                    label_dict_train[attr] = label_list
                
                for attr in tqdm(all_attrs, desc="Training initial models", ncols=120):
                    attr_name, model, _, _, _, _ = train_model(
                        attr, feat_dict_train[attr], label_dict_train[attr], num_epochs
                    )
                    if model is not None:
                        model_col[attr] = model
                
                logger.info(f"成功训练 {len(model_col)} 个模型")
            total_time += t.duration
            
            # 将训练集样本的标签作为第一次预测结果放入model_prediction_history
            # 因为用这些样本训练的模型对它们的预测结果应该与标签一致
            for attr in all_attrs:
                # 正确样本（标签为0）的预测概率设为0.0
                for idx, value in train_data_dict[attr]['right']:
                    model_prediction_history[attr][idx].append(0.0)
                # 错误样本（标签为1）的预测概率设为1.0
                for idx, value in train_data_dict[attr]['wrong']:
                    model_prediction_history[attr][idx].append(1.0)
            
            logger.info(f"已将训练集样本的标签作为初始预测结果加入预测历史")
            
            # ==================== 步骤8: 迭代优化 ====================
            logger.info(f"开始迭代优化，共 {ITERATIONS} 轮")
            
            for iteration in range(ITERATIONS):
                logger.info(f"\n{'='*50} 迭代 {iteration + 1}/{ITERATIONS} {'='*50}")
                para_file.write(f"\n--- Iteration {iteration + 1} ---\n")
                
                # 初始化本轮迭代的训练集变化记录
                if RESULT_ANALYZE:
                    iteration_train_changes = defaultdict(dict)
                
                # ========== 8.1 选择最优聚类 ==========
                with Timer(f'Iteration {iteration+1} - Select Optimal Cluster', logger, time_file) as t:
                    indices_dict = process_select_optimal_cluster(
                        train_data_dict, cluster_index_dict, dirty_csv, all_attrs, related_attrs_dict,
                        pre_funcs_for_attr, resp_path, logger, 
                        residual_method='both', previously_selected_clusters=previously_selected_clusters,
                        cluster_selection_window=CLUSTER_SELECTION_WINDOW
                    )
                    
                    # 记录已选择的聚类
                    for attr, cluster_info in indices_dict.items():
                        if attr in cluster_index_dict:
                            for cluster_idx, cluster_indices in enumerate(cluster_index_dict[attr][1:]):
                                if set(cluster_indices) == set(indices_dict.get(attr, [])):
                                    previously_selected_clusters[attr].append(cluster_idx)
                                    break
                total_time += t.duration

                # ========== 8.2 过滤已在训练集中的索引 ==========
                # 过滤掉已经在训练集中的索引
                filtered_indices_dict = {}
                already_in_train = 0
                
                for attr, indices in indices_dict.items():
                    train_indices = set()
                    # 收集该属性在训练集中的所有索引
                    for idx, _ in train_data_dict.get(attr, {}).get('right', []):
                        train_indices.add(idx)
                    for idx, _ in train_data_dict.get(attr, {}).get('wrong', []):
                        train_indices.add(idx)
                    
                    # 过滤掉已在训练集中的索引
                    unlabeled_indices = [idx for idx in indices if idx not in train_indices]
                    filtered_indices_dict[attr] = unlabeled_indices
                    already_in_train += len(indices) - len(unlabeled_indices)
                
                current_to_label = sum(len(indices) for indices in filtered_indices_dict.values())
                logger.info(f"本轮需要标注的样本数: {current_to_label} (已过滤 {already_in_train} 个训练集中的样本)")
                
                if current_to_label == 0:
                    logger.info("没有新的样本需要标注，跳过本轮")
                    continue
                
                labeled_number += current_to_label
                
                # ========== 8.3 LLM标注新样本 ==========
                with Timer(f'Iteration {iteration+1} - LLM Labeling', logger, time_file) as t:
                    for attr_name, indices in filtered_indices_dict.items():
                        if len(indices) == 0:
                            continue
                        
                        # 获取该列的标准模式（如果有）
                        attr_canonical_patterns = canonical_patterns_dict.get(attr_name, None)
                        
                        result = llm_label_indices(
                            attr_name, indices, dirty_csv, related_attrs_dict,
                            high_confidence_right_dict, high_confidence_wrong_dict,
                            error_checking_res_directory, err_check_val_num_per_query,
                            canonical_patterns=attr_canonical_patterns
                        )
                        
                        # 累积到历史标注
                        for idx, value, label in result.get(attr_name, []):
                            index_value_label_history[attr_name][idx].append(label)
                total_time += t.duration
                
                # ========== 8.4 模型预测并获取概率 ==========
                with Timer(f'Iteration {iteration+1} - Model Prediction', logger, time_file) as t:
                    det_wrong_list_res = []
                    all_predictions = {}  # {attr: [(idx, attr, pred_label, pred_proba), ...]}
                    
                    for col, attr in enumerate(all_attrs):
                        if attr not in model_col:
                            continue
                        
                        # 获取该列的标准模式（如果有）
                        attr_canonical_patterns = canonical_patterns_dict.get(attr, None)
                        predictions = make_predictions_with_proba(
                            col, attr, dirty_csv, model_col, related_attrs_dict,
                            funcs_for_attr, feature_all_dict, resp_path,
                            canonical_patterns=attr_canonical_patterns
                        )
                        all_predictions[attr] = predictions
                        
                        # 记录预测历史
                        for idx, attr_name, pred_label, pred_proba in predictions:
                            model_prediction_history[attr_name][idx].append(pred_proba)
                            if pred_label == 1:
                                det_wrong_list_res.append((idx, attr_name))
                total_time += t.duration
                
                # ========== 8.5 计算置信度并更新训练集 ==========
                with Timer(f'Iteration {iteration+1} - Confidence Calculation', logger, time_file) as t:
                    alpha = 0.7
                    beta = 1.2
                    
                    # 比较LLM和分类器标注
                    comparison_results = compare_llm_and_classifier_labels(
                        index_value_label_history, det_wrong_list_res, dirty_csv, related_attrs_dict
                    )
                    
                    new_high_conf_samples = 0
                    
                    for attr in all_attrs:
                        related_attrs = list(related_attrs_dict[attr])
                        
                        for idx in filtered_indices_dict.get(attr, []):
                            # 改进的置信度计算公式：
                            # 综合考虑分类器置信度、LLM一致性、标注次数和标注一致情况
                            # 
                            # 公式设计思路：
                            # 1. 当MLP和LLM标注一致时：主要依赖分类器置信度和LLM一致性
                            # 2. 当MLP和LLM标注不一致时：需要LLM标注次数足够多且一致性高才能信任
                            # 
                            # 统一公式：
                            # Conf = w_cls * C_cls + w_llm * C_llm + w_agree * I_agree + w_count * C_count
                            # 
                            # 其中：
                            # - C_cls = |p - 0.5| * 2  (分类器置信度，归一化到[0,1])
                            # - C_llm = LLM一致性 (范围[0,1])
                            # - I_agree = 1 if MLP和LLM一致 else 0
                            # - C_count = min(标注次数/5, 1)  (标注次数归一化，5次为满分)
                            # 
                            # 权重设置：
                            # - w_cls = 0.3  (分类器置信度权重)
                            # - w_llm = 0.4  (LLM一致性权重，最重要)
                            # - w_agree = 0.2  (标注一致性权重)
                            # - w_count = 0.1  (标注次数权重)
                            # 
                            # 这样设计的好处：
                            # 1. 当标注一致(I_agree=1)且分类器置信度高时，即使标注次数少也能达到阈值
                            # 2. 当标注不一致(I_agree=0)时，需要LLM标注次数多(C_count高)且一致性高(C_llm高)才能补偿
                            
                            # 获取LLM标注历史
                            label_history = index_value_label_history[attr].get(idx, [])
                            llm_consistency, llm_majority_label = calculate_llm_consistency(label_history)
                            llm_label_count = len(label_history)
                            
                            # 获取模型预测概率
                            pred_proba = 0.5
                            if attr in all_predictions:
                                for pred_idx, pred_attr, pred_label, p in all_predictions[attr]:
                                    if pred_idx == idx:
                                        pred_proba = p
                                        break
                            
                            # 判断MLP和LLM是否一致
                            mlp_label = 1 if pred_proba >= 0.5 else 0
                            is_agree = 1 if mlp_label == llm_majority_label else 0
                            
                            # 计算各个置信度分量
                            classifier_confidence = abs(pred_proba - 0.5) * 2  # 归一化到[0,1]
                            count_confidence = min(llm_label_count / 5.0, 1.0)  # 5次标注为满分
                            
                            # 权重参数
                            w_cls = 0.1
                            w_llm = 0.4
                            w_agree = 0.2
                            w_count = 0.3
                            
                            # 综合置信度计算
                            confidence = (w_cls * classifier_confidence + 
                                        w_llm * llm_consistency + 
                                        w_agree * is_agree + 
                                        w_count * count_confidence) # 0.7
                            
                            value = dirty_csv.loc[idx, [attr] + related_attrs].to_dict()
                            
                            # 根据置信度分类：高于高置信度阈值的加入训练数据
                            if confidence >= TRAIN_HIGH_CONFIDENCE_THRESHOLD:
                                # 构建置信度计算公式字符串
                                conf_formula = f"{w_cls}*{classifier_confidence:.4f} + {w_llm}*{llm_consistency:.4f} + {w_agree}*{is_agree} + {w_count}*{count_confidence:.4f} = {confidence:.4f}"
                                
                                if llm_majority_label == 1:
                                    if (idx, value) not in train_data_dict[attr]['wrong']:
                                        train_data_dict[attr]['wrong'].append((idx, value))
                                        new_high_conf_samples += 1
                                        # 记录新加入的样本
                                        if RESULT_ANALYZE:
                                            if 'added_wrong' not in iteration_train_changes[attr]:
                                                iteration_train_changes[attr]['added_wrong'] = []
                                            iteration_train_changes[attr]['added_wrong'].append({
                                                'idx': int(idx),
                                                'value': value,
                                                'confidence': confidence,
                                                'confidence_formula': conf_formula,
                                                'llm_consistency': llm_consistency,
                                                'classifier_confidence': classifier_confidence,
                                                'is_agree': is_agree,
                                                'count_confidence': count_confidence
                                            })
                                else:
                                    if (idx, value) not in train_data_dict[attr]['right']:
                                        train_data_dict[attr]['right'].append((idx, value))
                                        new_high_conf_samples += 1
                                        # 记录新加入的样本
                                        if RESULT_ANALYZE:
                                            if 'added_right' not in iteration_train_changes[attr]:
                                                iteration_train_changes[attr]['added_right'] = []
                                            iteration_train_changes[attr]['added_right'].append({
                                                'idx': int(idx),
                                                'value': value,
                                                'confidence': confidence,
                                                'confidence_formula': conf_formula,
                                                'llm_consistency': llm_consistency,
                                                'classifier_confidence': classifier_confidence,
                                                'is_agree': is_agree,
                                                'count_confidence': count_confidence
                                            })

                    
                    logger.info(f"新增高置信度样本: {new_high_conf_samples}")
                    para_file.write(f"New high conf samples: {new_high_conf_samples}\n")
                total_time += t.duration
                
                # ========== 8.6 重新训练模型 ==========
                # 只有在训练数据增加时才重新训练模型
                if new_high_conf_samples > 0:
                    with Timer(f'Iteration {iteration+1} - Model Retraining', logger, time_file) as t:
                        feat_dict_train = {}
                        label_dict_train = {}
                        
                        for attr in all_attrs:
                            # 获取该列的标准模式（如果有）
                            attr_canonical_patterns = canonical_patterns_dict.get(attr, None)
                            attr_name, feature_list, label_list = process_attr_train_feat(
                                attr, dirty_csv, train_data_dict, related_attrs_dict,
                                funcs_for_attr, feature_all_dict, resp_path,
                                canonical_patterns=attr_canonical_patterns
                            )
                            feat_dict_train[attr] = feature_list
                            label_dict_train[attr] = label_list
                        
                        for attr in tqdm(all_attrs, desc=f"Retraining models (iter {iteration+1})", ncols=120):
                            attr_name, model, _, _, _, _ = train_model(
                                attr, feat_dict_train[attr], label_dict_train[attr], num_epochs
                            )
                            if model is not None:
                                model_col[attr] = model
                    total_time += t.duration
                
                    # ========== 8.7 用新模型重新预测训练数据，计算稳定性置信度 ==========
                    with Timer(f'Iteration {iteration+1} - Stability Confidence', logger, time_file) as t:
                        alpha_stab = 0.7
                        low_conf_removed = 0
                        
                        # 对训练数据中的每个样本重新预测
                        for attr in all_attrs:
                            if attr not in model_col:
                                continue
                            
                            model = model_col[attr]
                            related_attrs = list(related_attrs_dict[attr])
                            
                            # 收集需要移除的低置信度样本，同时将样本分类到高/中置信度词典
                            samples_to_remove_right = []
                            samples_to_remove_wrong = []
                            
                            # 检查right样本
                            for idx, value in train_data_dict[attr]['right']:
                                # 获取预测历史
                                pred_history = model_prediction_history[attr].get(idx, [])
                                
                                if len(pred_history) >= 2:
                                    stability = calculate_prediction_stability(pred_history)
                                    current_proba = pred_history[-1] if pred_history else 0.5
                                    
                                    # 计算训练数据置信度
                                    train_confidence = calculate_training_confidence(current_proba, stability, alpha_stab)
                                    
                                    # 构建train_confidence计算公式字符串
                                    normalized_pred_conf = 2 * abs(current_proba - 0.5)
                                    train_conf_formula = f"{alpha_stab}*2*|{current_proba:.4f}-0.5| + {1-alpha_stab}*{stability:.4f} = {alpha_stab}*{normalized_pred_conf:.4f} + {1-alpha_stab}*{stability:.4f} = {train_confidence:.4f}"
                                    
                                    # 根据置信度分类
                                    if train_confidence < MID_CONFIDENCE_THRESHOLD:
                                        # 低置信度样本移除
                                        samples_to_remove_right.append((idx, value))
                                        low_conf_removed += 1
                                        # 记录移除的样本
                                        if RESULT_ANALYZE:
                                            if 'removed_right' not in iteration_train_changes[attr]:
                                                iteration_train_changes[attr]['removed_right'] = []
                                            iteration_train_changes[attr]['removed_right'].append({
                                                'idx': int(idx),
                                                'value': value,
                                                'train_confidence': train_confidence,
                                                'train_confidence_formula': train_conf_formula,
                                                'stability': stability,
                                                'current_proba': current_proba
                                            })
                                    elif train_confidence >= HIGH_CONFIDENCE_THRESHOLD:
                                        # 高置信度样本加入高置信度词典（去重）
                                        if value not in high_confidence_right_dict[attr]:
                                            high_confidence_right_dict[attr].append(value)
                                            # 记录新加入高置信度类的样本
                                            if RESULT_ANALYZE:
                                                if 'new_high_conf_right' not in iteration_train_changes[attr]:
                                                    iteration_train_changes[attr]['new_high_conf_right'] = []
                                                iteration_train_changes[attr]['new_high_conf_right'].append({
                                                    'idx': int(idx),
                                                    'value': value,
                                                    'train_confidence': train_confidence,
                                                    'train_confidence_formula': train_conf_formula,
                                                    'stability': stability,
                                                    'current_proba': current_proba
                                                })
                                    else:
                                        # 中等置信度样本加入中置信度词典（去重）
                                        if value not in mid_confidence_right_dict[attr]:
                                            mid_confidence_right_dict[attr].append(value)
                            
                            # 检查wrong样本
                            for idx, value in train_data_dict[attr]['wrong']:
                                pred_history = model_prediction_history[attr].get(idx, [])
                                
                                if len(pred_history) >= 2:
                                    stability = calculate_prediction_stability(pred_history)
                                    current_proba = pred_history[-1] if pred_history else 0.5
                                    
                                    train_confidence = calculate_training_confidence(current_proba, stability, alpha_stab)
                                    
                                    # 构建train_confidence计算公式字符串
                                    normalized_pred_conf = 2 * abs(current_proba - 0.5)
                                    train_conf_formula = f"{alpha_stab}*2*|{current_proba:.4f}-0.5| + {1-alpha_stab}*{stability:.4f} = {alpha_stab}*{normalized_pred_conf:.4f} + {1-alpha_stab}*{stability:.4f} = {train_confidence:.4f}"
                                    
                                    # 根据置信度分类
                                    if train_confidence < MID_CONFIDENCE_THRESHOLD:
                                        # 低置信度样本移除
                                        samples_to_remove_wrong.append((idx, value))
                                        low_conf_removed += 1
                                        # 记录移除的样本
                                        if RESULT_ANALYZE:
                                            if 'removed_wrong' not in iteration_train_changes[attr]:
                                                iteration_train_changes[attr]['removed_wrong'] = []
                                            iteration_train_changes[attr]['removed_wrong'].append({
                                                'idx': int(idx),
                                                'value': value,
                                                'train_confidence': train_confidence,
                                                'train_confidence_formula': train_conf_formula,
                                                'stability': stability,
                                                'current_proba': current_proba
                                            })
                                    elif train_confidence >= HIGH_CONFIDENCE_THRESHOLD:
                                        # 高置信度样本加入高置信度词典（去重）
                                        if value not in high_confidence_wrong_dict[attr]:
                                            high_confidence_wrong_dict[attr].append(value)
                                            # 记录新加入高置信度类的样本
                                            if RESULT_ANALYZE:
                                                if 'new_high_conf_wrong' not in iteration_train_changes[attr]:
                                                    iteration_train_changes[attr]['new_high_conf_wrong'] = []
                                                iteration_train_changes[attr]['new_high_conf_wrong'].append({
                                                    'idx': int(idx),
                                                    'value': value,
                                                    'train_confidence': train_confidence,
                                                    'train_confidence_formula': train_conf_formula,
                                                    'stability': stability,
                                                    'current_proba': current_proba
                                                })
                                    else:
                                        # 中等置信度样本加入中置信度词典（去重）
                                        if value not in mid_confidence_wrong_dict[attr]:
                                            mid_confidence_wrong_dict[attr].append(value)
                            
                            # 移除低置信度样本
                            for item in samples_to_remove_right:
                                if item in train_data_dict[attr]['right']:
                                    train_data_dict[attr]['right'].remove(item)
                            
                            for item in samples_to_remove_wrong:
                                if item in train_data_dict[attr]['wrong']:
                                    train_data_dict[attr]['wrong'].remove(item)
                        
                        logger.info(f"移除低置信度样本: {low_conf_removed}")
                        para_file.write(f"Removed low conf samples: {low_conf_removed}\n")
                else:
                    logger.info("训练数据未增加，跳过该轮模型训练和预测")
                total_time += t.duration
                
                # 统计当前训练集大小
                current_train_size = sum(
                    len(train_data_dict[attr]['right']) + len(train_data_dict[attr]['wrong'])
                    for attr in all_attrs
                )
                logger.info(f"当前训练集大小: {current_train_size}")
                para_file.write(f"Current training set size: {current_train_size}\n")

                # 根据配置选择是否计算每轮的F1分数
                if COMPUTE_F1_PER_ITERATION and len(model_col) > 0:
                    logger.info(f"计算第 {iteration + 1} 轮的F1分数...")
                    with Timer(f'Iteration {iteration+1} - Computing F1 Score', logger, time_file) as t:
                        precision, recall, f1, detected, total = compute_f1_score_for_iteration(
                            model_col, dirty_csv, clean_csv, all_attrs, related_attrs_dict,
                            funcs_for_attr, feature_all_dict, resp_path, logger,
                            canonical_patterns_dict=canonical_patterns_dict
                        )
                        
                        # 记录F1分数
                        f1_scores_per_iteration.append({
                            'iteration': iteration + 1,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'detected_errors': detected,
                            'total_errors': total,
                            'train_set_size': current_train_size
                        })
                        
                        para_file.write(f"Iteration {iteration + 1} F1: {f1:.4f} (P: {precision:.4f}, R: {recall:.4f})\n")
                    total_time += t.duration
                # 根据配置保存每轮的分析结果
                if RESULT_ANALYZE:
                    # 创建分析结果目录
                    analyze_dir = os.path.join(resp_path, 'iteration_analysis')
                    os.makedirs(analyze_dir, exist_ok=True)
                    
                    iteration_result = {
                        'iteration': iteration + 1,
                        'train_data': {},
                        'misclassified_samples': []
                    }
                    
                    # 保存当前轮次的训练数据
                    for attr in all_attrs:
                        iteration_result['train_data'][attr] = {
                            'right': [(int(idx), val) for idx, val in train_data_dict[attr]['right']],
                            'wrong': [(int(idx), val) for idx, val in train_data_dict[attr]['wrong']]
                        }
                    
                    # 找出分类器错误分类的样本（误报和漏报）
                    for attr in all_attrs:
                        if attr not in model_col:
                            continue
                        
                        related_attrs = list(related_attrs_dict[attr])
                        
                        # 检查预测结果中的误报（预测为错误但实际正确）
                        for idx, pred_attr in det_wrong_list_res:
                            if pred_attr == attr:
                                dirty_val = str(dirty_csv.loc[idx, attr])
                                clean_val = str(clean_csv.loc[idx, attr])
                                if dirty_val == clean_val:  # 误报
                                    iteration_result['misclassified_samples'].append({
                                        'type': 'false_positive',
                                        'attr': attr,
                                        'idx': int(idx),
                                        'value': dirty_csv.loc[idx, [attr] + related_attrs].to_dict(),
                                        'dirty_val': dirty_val,
                                        'clean_val': clean_val
                                    })
                        
                        # 检查漏报（实际错误但预测为正确）
                        detected_indices = {idx for idx, pred_attr in det_wrong_list_res if pred_attr == attr}
                        for idx in range(len(dirty_csv)):
                            dirty_val = str(dirty_csv.loc[idx, attr])
                            clean_val = str(clean_csv.loc[idx, attr])
                            if dirty_val != clean_val and idx not in detected_indices:  # 漏报
                                iteration_result['misclassified_samples'].append({
                                    'type': 'false_negative',
                                    'attr': attr,
                                    'idx': int(idx),
                                    'value': dirty_csv.loc[idx, [attr] + related_attrs].to_dict(),
                                    'dirty_val': dirty_val,
                                    'clean_val': clean_val
                                })
                    
                    # 保存到文件
                    iteration_file = os.path.join(analyze_dir, f'iteration_{iteration + 1}_analysis.json')
                    with open(iteration_file, 'w', encoding='utf-8') as f:
                        json.dump(iteration_result, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"迭代 {iteration + 1} 分析结果已保存到: {iteration_file}")
                    logger.info(f"  误分类样本数: {len(iteration_result['misclassified_samples'])}")
                    
                    # 保存训练集变化到文件
                    train_change_dir = os.path.join(resp_path, 'train_set_changes')
                    os.makedirs(train_change_dir, exist_ok=True)
                    
                    iteration_changes = {
                        'iteration': iteration + 1,
                        'changes': dict(iteration_train_changes)
                    }
                    
                    changes_file = os.path.join(train_change_dir, f'iteration_{iteration + 1}_train_changes.json')
                    with open(changes_file, 'w', encoding='utf-8') as f:
                        json.dump(iteration_changes, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
                    
                    # 统计变化数量
                    total_added = sum(len(iteration_train_changes[attr].get('added_right', [])) + len(iteration_train_changes[attr].get('added_wrong', [])) for attr in iteration_train_changes)
                    total_removed = sum(len(iteration_train_changes[attr].get('removed_right', [])) + len(iteration_train_changes[attr].get('removed_wrong', [])) for attr in iteration_train_changes)
                    total_new_high_conf = sum(len(iteration_train_changes[attr].get('new_high_conf_right', [])) + len(iteration_train_changes[attr].get('new_high_conf_wrong', [])) for attr in iteration_train_changes)
                    
                    logger.info(f"  训练集变化已保存到: {changes_file}")
                    logger.info(f"  新加入训练集: {total_added}, 移除: {total_removed}, 新高置信度: {total_new_high_conf}")

            # ==================== 步骤9: 根据高置信度样本生成函数 ====================
            logger.info("根据高置信度样本生成函数")
            with Timer('Generating Functions from High Confidence Samples', logger, time_file) as t:
                err_gen_dict, funcs_for_attr = process_gen_err_funcs(
                    FUNC_USE, resp_path, funcs_directory, dirty_csv, all_attrs,
                    para_file, related_attrs_dict, high_confidence_right_dict, high_confidence_wrong_dict, API_USE, MODEL_TYPE
                )
            total_time += t.duration
            
            # ==================== 步骤10: 最终模型训练（使用生成的函数） ====================
            if FUNC_USE and funcs_for_attr:
                logger.info("使用生成的函数重新训练最终模型")
                with Timer('Final Model Training with Functions', logger, time_file) as t:
                    feat_dict_train = {}
                    label_dict_train = {}
                    
                    for attr in all_attrs:
                        # 获取该列的标准模式（如果有）
                        attr_canonical_patterns = canonical_patterns_dict.get(attr, None)
                        attr_name, feature_list, label_list = process_attr_train_feat(
                            attr, dirty_csv, train_data_dict, related_attrs_dict,
                            funcs_for_attr, feature_all_dict, resp_path,
                            canonical_patterns=attr_canonical_patterns
                        )
                        feat_dict_train[attr] = feature_list
                        label_dict_train[attr] = label_list
                    
                    for attr in tqdm(all_attrs, desc="Training final models", ncols=120):
                        attr_name, model, _, _, _, _ = train_model(
                            attr, feat_dict_train[attr], label_dict_train[attr], num_epochs
                        )
                        if model is not None:
                            model_col[attr] = model
                total_time += t.duration
            
            # ==================== 步骤11: 评估LLM标注 ====================
            logger.info("评估LLM标注结果")
            with Timer('Evaluating LLM Labeling', logger, time_file) as t:
                # 转换历史标注为最终标签格式
                _, final_labels = convert_label_history_to_train_data(
                    index_value_label_history, dirty_csv, related_attrs_dict,
                    0.0, all_attrs  # 使用0阈值获取所有标签
                )
                measure_status = measure_llm_label(
                    resp_path, clean_csv, all_attrs, related_attrs_dict, gt_wrong_dict, final_labels
                )
            total_time += t.duration
            
            # ==================== 步骤12: 标签扩散 ====================
            if LABEL_PROP:
                logger.info("执行标签扩散")
                with Timer('Label Propagation', logger, time_file) as t:
                    det_wrong_list, det_right_list = label_prop(
                        resp_path, dirty_path, clean_path, cluster_index_dict, final_labels, LABEL_PROP
                    )
                total_time += t.duration
            else:
                logger.info("标签扩散已禁用")
            
            # ==================== 步骤13: 最终预测 ====================
            logger.info("使用最终模型进行错误检测")
            
            # 重新加载特征缓存
            if os.path.exists(os.path.join(resp_path, 'cluster_feat_dict.pkl')):
                with open(os.path.join(resp_path, 'cluster_feat_dict.pkl'), 'rb') as f:
                    feature_all_dict = pickle.load(f)
            
            det_wrong_list_res = []
            with Timer('Final Prediction', logger, time_file) as t:
                for col, attr in tqdm(enumerate(all_attrs), desc="Making final predictions", ncols=120):
                    # 获取该列的标准模式（如果有）
                    attr_canonical_patterns = canonical_patterns_dict.get(attr, None)
                    wrong_cells = make_predictions(
                        col, attr, dirty_csv, model_col, related_attrs_dict,
                        funcs_for_attr, feature_all_dict, resp_path,
                        canonical_patterns=attr_canonical_patterns
                    )
                    for cell in wrong_cells:
                        if cell not in det_wrong_list_res:
                            det_wrong_list_res.append(cell)
            total_time += t.duration
            
            # ==================== 步骤14: 评估检测结果 ====================
            logger.info("评估错误检测结果")
            det_res_path = os.path.join(resp_path, "func_det_res.txt")
            measure_detect(clean_path, dirty_path, list(det_wrong_list_res), det_res_path)
            
            # 打印预测错误详情
            print_prediction_errors(
                dirty_csv, clean_csv, det_wrong_list_res, all_attrs, 
                related_attrs_dict, logger, resp_path
            )
            
            # ==================== 步骤15: 保存结果 ====================
            logger.info("保存结果文件")
            
            # 保存高置信度和中等置信度样本
            save_confidence_samples(
                {'right': high_confidence_right_dict, 'wrong': high_confidence_wrong_dict},
                {'right': mid_confidence_right_dict, 'wrong': mid_confidence_wrong_dict},
                resp_path
            )
            
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
            
            # 保存每轮F1分数（如果启用）
            if COMPUTE_F1_PER_ITERATION and len(f1_scores_per_iteration) > 0:
                f1_history_path = os.path.join(resp_path, 'f1_scores_per_iteration.json')
                with open(f1_history_path, 'w', encoding='utf-8') as f:
                    json.dump(f1_scores_per_iteration, f, ensure_ascii=False, indent=2)
                logger.info(f"F1分数历史已保存到: {f1_history_path}")
            
            # ==================== 完成 ====================
            time_end = time.time()
            total_time += time_end - time_start
            
            para_file.write(f"\nTotal LLM labeled samples: {labeled_number}\n")
            para_file.write(f"Final training set size: {sum(len(train_data_dict[attr]['right']) + len(train_data_dict[attr]['wrong']) for attr in all_attrs)}\n")
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
