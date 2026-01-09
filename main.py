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
                        guide_gen_prompt, 
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
                                 funcs_for_attr, feature_all_dict, resp_path):
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
        result = single_val_feat(cell_val, None, funcs_for_attr, attr, idx, columns, feature_all_dict, resp_path)
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
                                    funcs_for_attr, feature_all_dict, resp_path, logger):
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
        
        wrong_cells = make_predictions(
            col, attr, dirty_csv, model_col, related_attrs_dict,
            funcs_for_attr, feature_all_dict, resp_path
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
                      error_checking_res_directory, err_check_val_num_per_query=20):
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
    
    for response, indices in responses_with_indices:
        content = response.replace('\\+', '').replace('\\n', '\n')
        
        wrong_pattern = err_pat_in_text_attr(attr_name)
        right_pattern = right_pat_in_text_attr(attr_name)
        
        events = []
        
        for m in re.finditer(wrong_pattern, content):
            text = normalize_string(
                m.group(1).replace("':'", "': '").replace(',', ', ').replace(',  ', ', ').replace('"', "'")
            ).replace('"{', '{', 1)[:-1] if m.group(1).startswith('"{') else normalize_string(m.group(1))
            events.append((m.start(), text, 1))
        
        for m in re.finditer(right_pattern, content):
            text = normalize_string(
                m.group(1).replace("':'", "': '").replace(',', ', ').replace(',  ', ', ').replace('"', "'")
            ).replace('"{', '{', 1)[:-1] if m.group(1).startswith('"{') else normalize_string(m.group(1))
            events.append((m.start(), text, 0))
        
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
                            funcs_for_attr, feature_all_dict, resp_path):
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
                                  list(dirty_csv.columns), feature_all_dict, resp_path)
        if feature:
            feature_list.append(feature)
            label_list.append(0)
    
    for idx, val in tqdm(wrong_samples, ncols=120, desc=f"Processing {attr} wrong values"):
        feature = single_val_feat(val, fasttext_model, funcs_for_attr, attr, -1, 
                                  list(dirty_csv.columns), feature_all_dict, resp_path)
        if feature:
            feature_list.append(feature)
            label_list.append(1)
    
    return attr, feature_list, label_list


def single_val_feat(val, fasttext_m, funcs_for_attr, attr, idx, all_attrs, feature_all_dict, resp_path):
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
        return idx, feature


def make_predictions(col, attr, dirty_csv, model_col, related_attrs_dict, funcs_for_attr, feature_all_dict, resp_path):
    if attr not in model_col.keys():
        return []
    model = model_col[attr]
    related_attrs = list(related_attrs_dict[attr])
    columns = list(dirty_csv.columns)
    
    results = []
    for idx in range(len(dirty_csv)):
        cell_val = dirty_csv.loc[idx, [attr]+related_attrs].to_dict()
        result = single_val_feat(cell_val, None, funcs_for_attr, attr, idx, columns, feature_all_dict, resp_path)
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
                        result = llm_label_indices(
                            attr_name, indices, dirty_csv, related_attrs_dict,
                            high_confidence_right_dict, high_confidence_wrong_dict,
                            error_checking_res_directory, err_check_val_num_per_query
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
                    attr_name, feature_list, label_list = process_attr_train_feat(
                        attr, dirty_csv, train_data_dict, related_attrs_dict,
                        funcs_for_attr, feature_all_dict, resp_path
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
                        
                        result = llm_label_indices(
                            attr_name, indices, dirty_csv, related_attrs_dict,
                            high_confidence_right_dict, high_confidence_wrong_dict,
                            error_checking_res_directory, err_check_val_num_per_query
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
                        
                        predictions = make_predictions_with_proba(
                            col, attr, dirty_csv, model_col, related_attrs_dict,
                            funcs_for_attr, feature_all_dict, resp_path
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
                            attr_name, feature_list, label_list = process_attr_train_feat(
                                attr, dirty_csv, train_data_dict, related_attrs_dict,
                                funcs_for_attr, feature_all_dict, resp_path
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
                            funcs_for_attr, feature_all_dict, resp_path, logger
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
                        attr_name, feature_list, label_list = process_attr_train_feat(
                            attr, dirty_csv, train_data_dict, related_attrs_dict,
                            funcs_for_attr, feature_all_dict, resp_path
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
                    wrong_cells = make_predictions(
                        col, attr, dirty_csv, model_col, related_attrs_dict,
                        funcs_for_attr, feature_all_dict, resp_path
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
