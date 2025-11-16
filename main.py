import argparse
import ast
import json
import multiprocessing
import os
import pickle
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from scipy import stats

import fasttext.util
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import yaml
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from distri_analys import LLMDataDistrAnalyzer
from feature import cluster, feat_gen_df, feat_gen_df_incremental, feat_gen_global_cache
from get_rel_attrs import (cal_all_column_nmi, cal_strong_res_column_nmi)
from measure import measure_detect
from prompt_gen import (create_err_gen_inst_prompt, err_clean_func_prompt,
                        error_check_prompt, pre_func_prompt,
                        create_clean_gen_inst_prompt, create_dirty_gen_inst_prompt,
                        guide_gen_prompt, 
                        )
from utility import (Logger, Timer, copy_file, copy_read_files_in_dir,
                     default_dict_of_lists, get_ans_from_llm, query_base,
                     rag_query, split_list_to_sublists, get_read_paths)


def subtask_det_initial(val_list, attr_name, indices):
    """
    处理LLM标注任务，针对指定的indices
    
    Args:
        val_list: 值列表
        attr_name: 属性名
        indices: 数据集中的索引列表，用于保存标注结果时关联
    
    Returns:
        LLM响应
    """
    str_list = [str(a_val) for a_val in val_list]
    vals_str = '\n'.join(str_list)
    prompt = error_check_prompt(vals_str, attr_name)
    if GUIDE_USE:
        response = rag_query(prompt, guide_content[attr_name])
    else:
        response = query_base(prompt)
    
    # 保存提示和响应，同时保存indices信息
    error_check_prompt_file = open(os.path.join(error_checking_res_directory, f'prompt_error_checking_{attr_name}.txt'), 'a', encoding='utf-8')
    error_check_prompt_file.write(prompt + '\n\n')
    error_check_prompt_file.close()
    
    # 保存响应和indices信息，使用追加模式
    error_checking_file = open(os.path.join(error_checking_res_directory, f'error_checking_{attr_name}.txt'), 'a', encoding='utf-8')
    # 在响应前添加indices信息，格式为: // indices: [idx1, idx2, ...]
    error_checking_file.write(f"// indices: {indices}\n")
    error_checking_file.write(response + '\n\n')
    error_checking_file.close()
    
    return response


def extract_func(text_content):
    try:
        code_blocks = re.findall(r'```(.*?)```', text_content, re.DOTALL)
    except re.error as e:
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
                print("Function name not found in the function definition.")
                continue
            if 'is_clean' in function_name:
                clean_func_list.append(function)
            elif 'is_dirty' in function_name:
                dirty_func_list.append(function)
    return clean_func_list, dirty_func_list


def extract_err_info(text, attr):
    information = []
    attr_name = attr
    lines = text.split('\n')
    for line in lines:
        err_info = []
        match = re.search(r'\[(.*?)\]', line)
        if match:
            try:
                data = match.group().replace("Reason: '", "'Reason: ")
                parsed_data = ast.literal_eval(data)
                err_info.append(attr_name)
                for i, content in enumerate(parsed_data):
                    if i != len(parsed_data) - 1 and i != 0:
                        err_info.append(str(content))
                    elif i == len(parsed_data) - 1:
                        err_info.append(content)
            except Exception as e:
                print("\n\nWhen processing error_err_info():" + line + "--" + attr)
                print(e)
        information.append(err_info)
    information = list(filter(None, information))
    return information


def extract_enhanced_info(text, attr):
    """
    增强版错误信息提取函数，能够处理包含特殊字符的复杂字符串
    解决原 extract_err_info 函数在解析包含单引号、连字符等特殊字符时失败的问题
    """
    information = []
    attr_name = attr
    lines = text.split('\n')
    for line in lines:
        err_info = []
        match = re.search(r'\[(.*?)\]', line)
        if match:
            try:
                data = match.group().replace("Reason: '", "'Reason: ")
                # 尝试使用 ast.literal_eval 解析
                try:
                    parsed_data = ast.literal_eval(data)
                except (SyntaxError, ValueError) as e:
                    # 如果 ast.literal_eval 失败，使用自定义解析逻辑
                    parsed_data = parse_complex_list(data)
                
                err_info.append(attr_name)
                for i, content in enumerate(parsed_data):
                    if i != len(parsed_data) - 1 and i != 0:
                        err_info.append(str(content))
                    elif i == len(parsed_data) - 1:
                        err_info.append(content)
            except Exception as e:
                print("\n\nWhen processing error_err_info_enhanced():" + line + "--" + attr)
                print(e)
        information.append(err_info)
    information = list(filter(None, information))
    return information


def parse_complex_list(data_str):
    """
    解析包含特殊字符的复杂列表字符串
    使用简单的方法：用单引号和逗号解析前两个元组，用大括号解析最后的词典结构
    """
    try:
        # 移除外层方括号
        content = data_str.strip('[]')
        
        # 初始化结果列表
        parsed_data = []
        
        # 查找字典部分的开始位置（第一个大括号）
        dict_start = content.find('{')
        
        if dict_start == -1:
            # 如果没有字典，使用简单的逗号分割
            parts = content.split(',')
            for part in parts:
                parsed_data.append(part.strip().strip('\'"'))
            return parsed_data
        
        # 分割非字典部分和字典部分
        non_dict_part = content[:dict_start].strip()
        dict_part = content[dict_start:].strip()
        
        # 处理非字典部分（前两个元组）
        if non_dict_part:
            # 使用正则表达式匹配单引号字符串
            pattern = r'\'([^\']*)\''
            matches = re.findall(pattern, non_dict_part)
            
            # 如果没有找到单引号字符串，尝试用逗号分割
            if not matches:
                parts = non_dict_part.split(',')
                for part in parts:
                    if part.strip():
                        parsed_data.append(part.strip())
            else:
                # 添加匹配到的字符串
                for match in matches:
                    if match.strip():
                        parsed_data.append(match.strip())
        
        # 处理字典部分
        if dict_part.startswith('{') and dict_part.endswith('}'):
            try:
                # 尝试使用json.loads解析字典（如果格式正确）
                try:
                    # 将单引号替换为双引号，以便json.loads可以解析
                    json_str = dict_part.replace("'", '"')
                    parsed_dict = json.loads(json_str)
                    parsed_data.append(parsed_dict)
                except json.JSONDecodeError:
                    # 如果json.loads失败，使用自定义解析逻辑
                    dict_content = dict_part[1:-1]  # 移除外层大括号
                    
                    # 分割键值对
                    result_dict = {}
                    
                    # 处理Reason和pattern description
                    reason_match = re.search(r'Reason:\s*([^,]+)', dict_content)
                    if reason_match:
                        result_dict['Reason'] = reason_match.group(1).strip()
                    
                    pattern_match = re.search(r'pattern description:\s*([^,}]+)', dict_content)
                    if pattern_match:
                        result_dict['pattern description'] = pattern_match.group(1).strip()
                    
                    # 处理其他键值对
                    # 使用正则表达式匹配键值对
                    kv_pattern = r'([^:]+):\s*([^,}]+)'
                    kv_matches = re.findall(kv_pattern, dict_content)
                    
                    for k, v in kv_matches:
                        k = k.strip().strip('\'"')
                        v = v.strip()
                        
                        # 跳过已经处理的Reason和pattern description
                        if k in ['Reason', 'pattern description']:
                            continue
                            
                        # 处理引号
                        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                            v = v[1:-1]
                        
                        result_dict[k] = v
                    
                    parsed_data.append(result_dict)
            except Exception as e:
                print(f"Error parsing dictionary: {e}")
                # 如果解析失败，尝试使用简单的键值对分割
                try:
                    dict_content = dict_part[1:-1]  # 移除外层大括号
                    # 分割字典键值对，处理包含单引号的值
                    kv_pairs = re.findall(r'([^:]+):\s*(\'[^\']*\'|"[^"]*"|[^,]+)', dict_content)
                    result_dict = {}
                    for k, v in kv_pairs:
                        k = k.strip().strip('\'"')
                        v = v.strip()
                        # 处理包含单引号的值，如 'O'Hare'
                        if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                            # 移除外层引号，保留内部引号
                            v = v[1:-1]
                        result_dict[k] = v
                    parsed_data.append(result_dict)
                except Exception as e2:
                    print(f"Error in fallback dictionary parsing: {e2}")
                    parsed_data.append(dict_part)
        
        return parsed_data
    except Exception as e:
        print(f"Error in parse_complex_list: {e}")
        # 如果所有方法都失败，返回原始字符串的简单分割
        try:
            # 最后的备选方案：简单分割并清理
            content = data_str.strip('[]')
            parts = content.split(',')
            return [part.strip().strip('\'"') for part in parts]
        except:
            return []


def gen_dirty_funcs(attr, clean_info, errs_info):
    dirty_str = "\n"
    clean_info = '\n'.join([str(i) for i in clean_info])
    try:
        dirty_str = dirty_str + '\n'.join([str(i) for i in errs_info])
    except Exception as e:
        print(f"Error: {e}\n When handling {errs_info}\n")
        dirty_str = dirty_str + str(errs_info)
        dirty_str = dirty_str + "\n"
    func_gen_prompt = err_clean_func_prompt(attr, clean_info, dirty_str)
    llm_gen_func = get_ans_from_llm(func_gen_prompt, api_use=API_USE)
    temp_clean_flist, dirty_flist = extract_func(llm_gen_func)
    return temp_clean_flist, dirty_flist, func_gen_prompt, llm_gen_func


def subtask_func_gen(attr_name, err_list, func_file_num, right_values_list):
    temp_clean_flist, dirty_flist, func_gen_prompt, llm_gen_func = gen_dirty_funcs(attr_name, right_values_list, err_list)
    funcs_for_attr = defaultdict(default_dict_of_lists)
    funcs_for_attr[attr_name]['clean'].extend(list(set(temp_clean_flist)))
    funcs_for_attr[attr_name]['dirty'].extend(list(set(dirty_flist)))
    with open(os.path.join(funcs_directory, f"prompt_funcs_zgen_{attr_name}{func_file_num}.txt"), 'w', encoding='utf-8') as prom_file:
        prom_file.write(func_gen_prompt)
    with open(os.path.join(funcs_directory, f"funcs_zgen_{attr_name}{func_file_num}.txt"), 'w', encoding='utf-8') as func_file:
        func_file.write("\n".join(list(set(temp_clean_flist))))
    return attr_name, funcs_for_attr


def process_gen_err_data(ERR_GEN_USE, ERR_GEN_READ, read_err_gen_path, err_gen_directory, dirty_csv, all_attrs, related_attrs_dict, center_index_value_label_dict, err_gen_dict, logger):
    if ERR_GEN_USE and ERR_GEN_READ:
        copy_read_files_in_dir(err_gen_directory, read_err_gen_path)
        for attr in all_attrs:
            if os.path.exists(os.path.join(err_gen_directory, f'err_gen_res_{attr}.txt')):
                with open(os.path.join(err_gen_directory, f'err_gen_res_{attr}.txt'), 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            err_dict = json.loads(line.strip())
                            err_gen_dict[attr]['dirty'].append(err_dict)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON for attribute {attr}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Unexpected error for attribute {attr}: {e}")
                            continue

    elif ERR_GEN_USE and not ERR_GEN_READ:
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(task_gen_err_data, attr, dirty_csv, center_index_value_label_dict, related_attrs_dict, err_gen_dict) for attr in all_attrs]
            outputs = [result.result() for result in results]

def prepare_enhanced_data_values(attr, dirty_csv, clean_csv, inconsistent_index_value_label_dict, related_attrs_dict, index_value_label_dict):
    """
    准备用于生成增强数据的 wrong_values、right_values 和 actual_right_values
    
    Args:
        attr: 属性名
        dirty_csv: 脏数据DataFrame
        clean_csv: 清洁数据DataFrame
        inconsistent_index_value_label_dict: 不一致的索引值标签字典
        related_attrs_dict: 相关属性字典
        index_value_label_dict: 索引值标签字典
    
    Returns:
        wrong_values: 错误值列表
        right_values: 正确值列表
        actual_right_values: 实际正确值列表
    """
    related_attrs = list(related_attrs_dict[attr])
    wrong_values = []
    right_values = []
    actual_right_values = []
    
    for idx, _, label in inconsistent_index_value_label_dict[attr]:
        if label == 1:
            # 与clean_csv对比，确定第一个值即dirty_csv.loc[int(idx), attr]是错误的还是正确的
            if dirty_csv.loc[int(idx), attr] != clean_csv.loc[int(idx), attr]:
                # 如果实际是错误的，则attr的脏值合并上related_attrs的干净值加入wrong_values
                wrong_values.append({attr: dirty_csv.loc[int(idx), attr], **clean_csv.loc[int(idx), related_attrs].to_dict()})
                right_values.append(clean_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            else:
                # 如果实际是正确的，则attr的干净值合并上related_attrs的干净值加入right_values
                actual_right_values.append(clean_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
                # 找到要删除的元素的索引
                for i, (stored_idx, stored_value, stored_label) in enumerate(index_value_label_dict[attr]):
                    if stored_idx == idx and stored_label == label:
                        # 找到匹配的元素，现在删除它
                        del index_value_label_dict[attr][i]
                        # 添加新元素
                        # 添加新元素，使用正确的字典格式
                        index_value_label_dict[attr].append((idx, dirty_csv.loc[int(idx), [attr] + related_attrs].to_dict(), 0))
                        break  # 找到后立即退出循环
        elif label == 0:
            if dirty_csv.loc[int(idx), attr] != clean_csv.loc[int(idx), attr]:
                wrong_values.append({attr: dirty_csv.loc[int(idx), attr], **clean_csv.loc[int(idx), related_attrs].to_dict()})
                right_values.append(clean_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
                # 找到要删除的元素的索引
                for i, (stored_idx, stored_value, stored_label) in enumerate(index_value_label_dict[attr]):
                    if stored_idx == idx and stored_label == label:
                        # 找到匹配的元素，现在删除它
                        del index_value_label_dict[attr][i]
                        # 添加新元素
                        # 添加新元素，使用正确的字典格式
                        index_value_label_dict[attr].append((idx, dirty_csv.loc[int(idx), [attr] + related_attrs].to_dict(), 0))
                        break  # 找到后立即退出循环
            else:
                actual_right_values.append(clean_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
    return wrong_values, right_values, actual_right_values

def prepare_wrong_right_values(attr, current_index_value_label_dict):
    """
    从当前索引值标签字典中提取 wrong_values 和 right_values
    
    Args:
        attr: 属性名
        current_index_value_label_dict: 当前索引值标签字典
    
    Returns:
        wrong_values: 错误值列表
        right_values: 正确值列表
    """
    wrong_values = []
    right_values = []
    
    for idx, value, label in current_index_value_label_dict[attr]:
        if label == 1:
            wrong_values.append(value)
        elif label == 0:
            right_values.append(value)
    
    return wrong_values, right_values

def process_gen_enhanced_data(ENHANCED_USE, ENHANCED_READ, read_enhanced_path, enhanced_data_directory, dirty_csv, clean_csv, all_attrs, related_attrs_dict, index_value_label_dict, current_index_value_label_dict, enhanced_gen_dict, num_gen, logger):
    if ENHANCED_USE and ENHANCED_READ:
        copy_read_files_in_dir(enhanced_data_directory, read_enhanced_path)
        for attr in all_attrs:
            if os.path.exists(os.path.join(enhanced_data_directory, f'clean_gen_res_{attr}.txt')):
                with open(os.path.join(enhanced_data_directory, f'clean_gen_res_{attr}.txt'), 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            clean_dict = json.loads(line.strip())
                            enhanced_gen_dict[attr]['clean'].append(clean_dict)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON for attribute {attr}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Unexpected error for attribute {attr}: {e}")
                            continue
            if os.path.exists(os.path.join(enhanced_data_directory, f'dirty_gen_res_{attr}.txt')):
                with open(os.path.join(enhanced_data_directory, f'dirty_gen_res_{attr}.txt'), 'r', encoding='utf-8') as file:
                    for line in file:
                        try:
                            err_dict = json.loads(line.strip())
                            enhanced_gen_dict[attr]['dirty'].append(err_dict)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON for attribute {attr}: {e}")
                            continue
                        except Exception as e:
                            logger.error(f"Unexpected error for attribute {attr}: {e}")
                            continue
            enhanced_gen_dict[attr]['dirty'] = random.sample(enhanced_gen_dict[attr]['dirty'], len(enhanced_gen_dict[attr]['clean']))

    elif ENHANCED_USE and not ENHANCED_READ:
        # 改为单线程处理，便于调试
        for attr in all_attrs:
            # 准备用于生成增强数据的值
            wrong_values, right_values, actual_right_values = prepare_enhanced_data_values(
                attr, dirty_csv, clean_csv, current_index_value_label_dict, related_attrs_dict, index_value_label_dict
            )
            
            # 调用生成增强数据的方法
            generate_enhanced_data_from_values(
                attr, dirty_csv, clean_csv, related_attrs_dict, enhanced_gen_dict, 
                wrong_values, right_values, actual_right_values, num_gen
            )

def generate_enhanced_data_from_values(attr, dirty_csv, clean_csv, related_attrs_dict, enhanced_gen_dict, 
                                      wrong_values, right_values, actual_right_values, num_gen):
    """
    根据提供的 wrong_values、right_values 和 actual_right_values 生成增强数据
    
    Args:
        attr: 属性名
        dirty_csv: 脏数据DataFrame
        clean_csv: 清洁数据DataFrame
        related_attrs_dict: 相关属性字典
        enhanced_gen_dict: 增强数据字典
        wrong_values: 错误值列表
        right_values: 正确值列表
        actual_right_values: 实际正确值列表
        num_gen: 生成数量
    """
    related_attrs = list(related_attrs_dict[attr])
    sep = "\n" + "="*40 + f" New run for attr: {attr} " + "="*40 + "\n\n"
    clean_gen_prompt_file = open(os.path.join(enhanced_gen_directory, f"prompt_ans_clean_gen_{attr}.txt"), 'a', encoding='utf-8')
    dirty_gen_prompt_file = open(os.path.join(enhanced_gen_directory, f"prompt_ans_dirty_gen_{attr}.txt"), 'a', encoding='utf-8')
    clean_gen_file = open(os.path.join(enhanced_gen_directory, f"clean_gen_{attr}.txt"), 'a', encoding='utf-8')
    dirty_gen_file = open(os.path.join(enhanced_gen_directory, f"dirty_gen_{attr}.txt"), 'a', encoding='utf-8')
    # 每次调用写入分隔符
    clean_gen_prompt_file.write(sep)
    dirty_gen_prompt_file.write(sep)
    clean_gen_file.write(sep)
    dirty_gen_file.write(sep)
    
    if len(right_values) == 0 and len(wrong_values) == 0 and len(actual_right_values) == 0:
        logger.warning(f"No data available for attribute {attr} to generate enhanced data.")
        return
            
            
    max_vals = 20
    if len(wrong_values) > max_vals:
        wrong_values_tmp = wrong_values[:max_vals]
    else:
        wrong_values_tmp = wrong_values
    if len(right_values) > max_vals:
        right_values_tmp = right_values[:max_vals]
    else:
        right_values_tmp = right_values
    if len(actual_right_values) > max_vals:
        actual_right_values_tmp = actual_right_values[:max_vals]
    else:
        actual_right_values_tmp = (actual_right_values + right_values_tmp)[:max_vals]
    
    # 创建变量来保存这次运行生成的增强数据
    new_clean_data = []
    new_dirty_data = []
    
    # 处理干净数据生成
    clean_gen_ans = ""
    if num_gen > 15:
        # 分批生成，每批20个
        num_batches = (num_gen + 14) // 15  # 向上取整
        for batch in range(num_batches):
            batch_size = min(15, num_gen - batch * 15)
            clean_gen_prompt = create_clean_gen_inst_prompt(actual_right_values_tmp, attr, num_gen=batch_size)
            batch_clean_ans = get_ans_from_llm(clean_gen_prompt, api_use=API_USE)
            clean_gen_ans += batch_clean_ans + "\n\n"
            clean_gen_prompt_file.write('*'*20 + f' batch {batch+1} prompt ' + '*'*20 + '\n' + clean_gen_prompt + '\n' + '*'*20 + f' batch {batch+1} answer ' + '*'*20 + '\n' + batch_clean_ans + '\n\n\n\n\n\n')
    else:
        # 一次性生成
        clean_gen_prompt = create_clean_gen_inst_prompt(actual_right_values_tmp, attr, num_gen=num_gen)
        clean_gen_ans = get_ans_from_llm(clean_gen_prompt, api_use=API_USE)
        clean_gen_prompt_file.write('*'*20 + ' prompt ' + '*'*20 + '\n' + clean_gen_prompt + '\n' + '*'*20 + ' answer ' + '*'*20 + '\n' + clean_gen_ans + '\n\n\n\n\n\n')
    clean_gen_prompt_file.close()
    clean_gen_file.write(clean_gen_ans)
    clean_gen_file.close()
    clean_info = extract_enhanced_info(clean_gen_ans, attr)

    if (len(wrong_values_tmp) ==0):
        logger.warning(f"No wrong values available for attribute {attr} to generate dirty data.")
        return
    # 处理干净数据，获取filtered_clean
    filtered_clean = []
    # filtered_clean.extend(right_values+actual_right_values)
    for clean in clean_info:
        if len(clean) < 4 or len(clean[3]) == 0 or not isinstance(clean[3], dict) or len(clean[3].keys()) < len([attr]+related_attrs) :
            continue

        try:
            if clean[0] in all_attrs and str(clean[-1]).strip() not in right_values and str(
                    clean[-1]).strip() not in wrong_values:
                enhanced_gen_dict[attr]['clean'].append(clean[3])
                filtered_clean.append(clean[3])
                # 添加到新数据列表
                new_clean_data.append(clean[3])
        except IndexError as e:
            logger.error(f"\nError: {e}\n Handling Value: {clean}\n Processing attribute: {attr}\n")
    
    # 处理脏数据生成（修改为对filtered_clean中的数据注入错误）
    dirty_gen_ans = ""
    num_dirty_to_generate = len(filtered_clean)  # 使用filtered_clean的长度作为需要生成的脏数据数量
    
    # 对filtered_clean分批，每批15个，注入错误
    if num_dirty_to_generate > 15:
        # 分批生成，每批15个
        num_batches = (num_dirty_to_generate + 14) // 15  # 向上取整
        for batch in range(num_batches):
            start_idx = batch * 15
            end_idx = min((batch + 1) * 15, num_dirty_to_generate)
            clean_for_error_injection = filtered_clean[start_idx:end_idx]
            
            dirty_gen_prompt = create_dirty_gen_inst_prompt(clean_for_error_injection, right_values_tmp, wrong_values_tmp, attr, num_errors=len(clean_for_error_injection))
            batch_dirty_ans = get_ans_from_llm(dirty_gen_prompt, api_use=API_USE)
            dirty_gen_ans += batch_dirty_ans + "\n\n"
            dirty_gen_prompt_file.write('*'*20 + f' batch {batch+1} prompt ' + '*'*20 + '\n' + dirty_gen_prompt + '\n' + '*'*20 + f' batch {batch+1} answer ' + '*'*20 + '\n' + batch_dirty_ans + '\n\n\n\n\n\n')
    else:
        # 一次性生成
        dirty_gen_prompt = create_dirty_gen_inst_prompt(filtered_clean, right_values_tmp, wrong_values_tmp, attr, num_errors=num_dirty_to_generate)
        dirty_gen_ans = get_ans_from_llm(dirty_gen_prompt, api_use=API_USE)
        dirty_gen_prompt_file.write('*'*20 + ' prompt ' + '*'*20 + '\n' + dirty_gen_prompt + '\n' + '*'*20 + ' answer ' + '*'*20 + '\n' + dirty_gen_ans + '\n\n\n\n\n\n')
    
    dirty_gen_prompt_file.close()
    dirty_gen_file.write(dirty_gen_ans)
    dirty_gen_file.close()
    dirty_info = extract_enhanced_info(dirty_gen_ans, attr)
    filtered_dirty = []
    filtered_dirty.extend(wrong_values)
    for dirty in dirty_info:
        try:
            if len(dirty) < 4 or len(dirty[3]) == 0 or not isinstance(dirty[3], dict) or len(dirty[3].keys()) < len([attr]+related_attrs) :
                continue
            if dirty[0] in all_attrs and str(dirty[-1]).strip() not in right_values and str(
                    dirty[-1]).strip() not in wrong_values:
                enhanced_gen_dict[attr]['dirty'].append(dirty[3])
                # 将字典格式的数据添加到filtered_dirty中，保持数据类型一致
                filtered_dirty.append(dirty[3])
                # 添加到新数据列表
                new_dirty_data.append(dirty[3])
        except IndexError as e:
            logger.error(f"\nError: {e}\n Handling Value: {dirty}\n Processing attribute: {attr}\n")
    
    # 只保存这次运行新生成的数据，而不是整个enhanced_gen_dict
    if new_clean_data:
        clean_gen_res_file = open(os.path.join(enhanced_gen_directory, f"clean_gen_res_{attr}.txt"), 'a', encoding='utf-8')
        for clean_dict in new_clean_data:
            json.dump(clean_dict, clean_gen_res_file)
            clean_gen_res_file.write('\n')
        clean_gen_res_file.close()
    
    if new_dirty_data:
        dirty_gen_res_file = open(os.path.join(enhanced_gen_directory, f"dirty_gen_res_{attr}.txt"), 'a', encoding='utf-8')
        for dirty_dict in new_dirty_data:
            json.dump(dirty_dict, dirty_gen_res_file)
            dirty_gen_res_file.write('\n')
        dirty_gen_res_file.close()

def generate_enhanced_data_from_wrong_right_values(attr, dirty_csv, clean_csv, related_attrs_dict, enhanced_gen_dict, 
                                                  wrong_values, right_values, num_gen):
    """
    根据提供的 wrong_values 和 right_values 生成增强数据
    专门用于第一次迭代，不与干净数据比较
    
    Args:
        attr: 属性名
        dirty_csv: 脏数据DataFrame
        clean_csv: 清洁数据DataFrame
        related_attrs_dict: 相关属性字典
        enhanced_gen_dict: 增强数据字典
        wrong_values: 错误值列表
        right_values: 正确值列表
        num_gen: 生成数量
    """
    related_attrs = list(related_attrs_dict[attr])
    sep = "\n" + "="*40 + f" New run for attr: {attr} " + "="*40 + "\n\n"
    clean_gen_prompt_file = open(os.path.join(enhanced_gen_directory, f"prompt_ans_clean_gen_{attr}.txt"), 'a', encoding='utf-8')
    dirty_gen_prompt_file = open(os.path.join(enhanced_gen_directory, f"prompt_ans_dirty_gen_{attr}.txt"), 'a', encoding='utf-8')
    clean_gen_file = open(os.path.join(enhanced_gen_directory, f"clean_gen_{attr}.txt"), 'a', encoding='utf-8')
    dirty_gen_file = open(os.path.join(enhanced_gen_directory, f"dirty_gen_{attr}.txt"), 'a', encoding='utf-8')
    # 每次调用写入分隔符
    clean_gen_prompt_file.write(sep)
    dirty_gen_prompt_file.write(sep)
    clean_gen_file.write(sep)
    dirty_gen_file.write(sep)
    
    if len(right_values) == 0 and len(wrong_values) == 0:
        logger.warning(f"No data available for attribute {attr} to generate enhanced data.")
        return
            
            
    max_vals = 20
    if len(wrong_values) > max_vals:
        wrong_values_tmp = wrong_values[:max_vals]
    else:
        wrong_values_tmp = wrong_values
    if len(right_values) > max_vals:
        right_values_tmp = right_values[:max_vals]
    else:
        right_values_tmp = right_values
    
    # 创建变量来保存这次运行生成的增强数据
    new_clean_data = []
    new_dirty_data = []
    
    # 处理干净数据生成 - 使用right_values生成干净数据
    clean_gen_ans = ""
    if num_gen > 15:
        # 分批生成，每批20个
        num_batches = (num_gen + 14) // 15  # 向上取整
        for batch in range(num_batches):
            batch_size = min(15, num_gen - batch * 15)
            clean_gen_prompt = create_clean_gen_inst_prompt(right_values_tmp, attr, num_gen=batch_size)
            batch_clean_ans = get_ans_from_llm(clean_gen_prompt, api_use=API_USE)
            clean_gen_ans += batch_clean_ans + "\n\n"
            clean_gen_prompt_file.write('*'*20 + f' batch {batch+1} prompt ' + '*'*20 + '\n' + clean_gen_prompt + '\n' + '*'*20 + f' batch {batch+1} answer ' + '*'*20 + '\n' + batch_clean_ans + '\n\n\n\n\n\n')
    else:
        # 一次性生成
        clean_gen_prompt = create_clean_gen_inst_prompt(right_values_tmp, attr, num_gen=num_gen)
        clean_gen_ans = get_ans_from_llm(clean_gen_prompt, api_use=API_USE)
        clean_gen_prompt_file.write('*'*20 + ' prompt ' + '*'*20 + '\n' + clean_gen_prompt + '\n' + '*'*20 + ' answer ' + '*'*20 + '\n' + clean_gen_ans + '\n\n\n\n\n\n')
    clean_gen_prompt_file.close()
    clean_gen_file.write(clean_gen_ans)
    clean_gen_file.close()
    clean_info = extract_enhanced_info(clean_gen_ans, attr)

    if (len(wrong_values_tmp) ==0):
        logger.warning(f"No wrong values available for attribute {attr} to generate dirty data.")
        return
    # 处理干净数据，获取filtered_clean
    filtered_clean = []
    for clean in clean_info:
        if len(clean) < 4 or len(clean[3]) == 0 or not isinstance(clean[3], dict) or len(clean[3].keys()) < len([attr]+related_attrs) :
            continue

        try:
            if clean[0] in all_attrs and str(clean[-1]).strip() not in right_values and str(
                    clean[-1]).strip() not in wrong_values:
                enhanced_gen_dict[attr]['clean'].append(clean[3])
                filtered_clean.append(clean[3])
                # 添加到新数据列表
                new_clean_data.append(clean[3])
        except IndexError as e:
            logger.error(f"\nError: {e}\n Handling Value: {clean}\n Processing attribute: {attr}\n")
    
    # 处理脏数据生成（修改为对filtered_clean中的数据注入错误）
    dirty_gen_ans = ""
    num_dirty_to_generate = len(filtered_clean)  # 使用filtered_clean的长度作为需要生成的脏数据数量
    
    # 对filtered_clean分批，每批15个，注入错误
    if num_dirty_to_generate > 15:
        # 分批生成，每批15个
        num_batches = (num_dirty_to_generate + 14) // 15  # 向上取整
        for batch in range(num_batches):
            start_idx = batch * 15
            end_idx = min((batch + 1) * 15, num_dirty_to_generate)
            clean_for_error_injection = filtered_clean[start_idx:end_idx]
            
            dirty_gen_prompt = create_dirty_gen_inst_prompt(clean_for_error_injection, right_values_tmp, wrong_values_tmp, attr, num_errors=len(clean_for_error_injection))
            batch_dirty_ans = get_ans_from_llm(dirty_gen_prompt, api_use=API_USE)
            dirty_gen_ans += batch_dirty_ans + "\n\n"
            dirty_gen_prompt_file.write('*'*20 + f' batch {batch+1} prompt ' + '*'*20 + '\n' + dirty_gen_prompt + '\n' + '*'*20 + f' batch {batch+1} answer ' + '*'*20 + '\n' + batch_dirty_ans + '\n\n\n\n\n\n')
    else:
        # 一次性生成
        dirty_gen_prompt = create_dirty_gen_inst_prompt(filtered_clean, right_values_tmp, wrong_values_tmp, attr, num_errors=num_dirty_to_generate)
        dirty_gen_ans = get_ans_from_llm(dirty_gen_prompt, api_use=API_USE)
        dirty_gen_prompt_file.write('*'*20 + ' prompt ' + '*'*20 + '\n' + dirty_gen_prompt + '\n' + '*'*20 + ' answer ' + '*'*20 + '\n' + dirty_gen_ans + '\n\n\n\n\n\n')
    
    dirty_gen_prompt_file.close()
    dirty_gen_file.write(dirty_gen_ans)
    dirty_gen_file.close()
    dirty_info = extract_enhanced_info(dirty_gen_ans, attr)
    filtered_dirty = []
    filtered_dirty.extend(wrong_values)
    for dirty in dirty_info:
        try:
            if len(dirty) < 4 or len(dirty[3]) == 0 or not isinstance(dirty[3], dict) or len(dirty[3].keys()) < len([attr]+related_attrs) :
                continue
            if dirty[0] in all_attrs and str(dirty[-1]).strip() not in right_values and str(
                    dirty[-1]).strip() not in wrong_values:
                enhanced_gen_dict[attr]['dirty'].append(dirty[3])
                # 将字典格式的数据添加到filtered_dirty中，保持数据类型一致
                filtered_dirty.append(dirty[3])
                # 添加到新数据列表
                new_dirty_data.append(dirty[3])
        except IndexError as e:
            logger.error(f"\nError: {e}\n Handling Value: {dirty}\n Processing attribute: {attr}\n")
    
    # 只保存这次运行新生成的数据，而不是整个enhanced_gen_dict
    if new_clean_data:
        clean_gen_res_file = open(os.path.join(enhanced_gen_directory, f"clean_gen_res_{attr}.txt"), 'a', encoding='utf-8')
        for clean_dict in new_clean_data:
            json.dump(clean_dict, clean_gen_res_file)
            clean_gen_res_file.write('\n')
        clean_gen_res_file.close()
    
    if new_dirty_data:
        dirty_gen_res_file = open(os.path.join(enhanced_gen_directory, f"dirty_gen_res_{attr}.txt"), 'a', encoding='utf-8')
        for dirty_dict in new_dirty_data:
            json.dump(dirty_dict, dirty_gen_res_file)
            dirty_gen_res_file.write('\n')
        dirty_gen_res_file.close()

def task_gen_err_data(attr, dirty_csv, index_value_label_dict, related_attrs_dict, err_gen_dict):
    related_attrs = list(related_attrs_dict[attr]) 
    err_gen_prompt_file = open(os.path.join(err_gen_directory, f"prompt_ans_error_gen_{attr}.txt"), 'w', encoding='utf-8')
    err_gen_file = open(os.path.join(err_gen_directory, f"error_gen_{attr}.txt"), 'w', encoding='utf-8')
    wrong_values = []
    right_values = []
    used_idx_list = {}
    for idx, _, label in index_value_label_dict[attr]:
        if label == 1:  
            wrong_values.append(dirty_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            used_idx_list[idx] = 1
        elif label == 0:  
            right_values.append(dirty_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            used_idx_list[idx] = 1 
    max_vals = 20
    if len(wrong_values) > max_vals:  
        wrong_values_tmp = wrong_values[:max_vals]
    else:
        wrong_values_tmp = wrong_values
    if len(right_values) > max_vals:  
        right_values_tmp = right_values[:max_vals]
    else:
        right_values_tmp = right_values
    err_gen_prompt = create_err_gen_inst_prompt(right_values_tmp, wrong_values_tmp, attr, num_errors=(len(right_values)))
    err_gen_ans = get_ans_from_llm(err_gen_prompt, api_use=API_USE)
    err_gen_prompt_file.write('*'*20 + ' prompt ' + '*'*20 + '\n' + err_gen_prompt + '\n' + '*'*20 + ' answer ' + '*'*20 + '\n' + err_gen_ans + '\n\n\n\n\n\n')
    err_gen_file.write(err_gen_ans)
    err_gen_prompt_file.close()
    err_gen_file.close()
    err_info = extract_err_info(err_gen_ans, attr)
    filtered_error = []
    filtered_error.extend(wrong_values)
    for err in err_info:
        try:
            if err[0] in all_attrs and str(err[-1]).strip() not in right_values and str(
                    err[-1]).strip() not in wrong_values:
                err_gen_dict[attr]['dirty'].append(err[3])
                filtered_error.extend([f"{err[3]}, {err[2]}"])
        except IndexError as e:
            logger.error(f"\nError: {e}\n Handling Value: {err}\n Processing attribute: {attr}\n")
    err_gen_res_file = open(os.path.join(err_gen_directory, f"err_gen_res_{attr}.txt"), 'w', encoding='utf-8')
    for err_dict in err_gen_dict[attr]['dirty']:
        json.dump(err_dict, err_gen_res_file)
        err_gen_res_file.write('\n')
    err_gen_res_file.close()

def task_gen_enhanced_data(attr, dirty_csv, clean_csv, index_value_label_dict, current_index_value_label_dict, related_attrs_dict, enhanced_gen_dict, num_gen):
    related_attrs = list(related_attrs_dict[attr])
    sep = "\n" + "="*40 + f" New run for attr: {attr} " + "="*40 + "\n\n"
    clean_gen_prompt_file = open(os.path.join(enhanced_gen_directory, f"prompt_ans_clean_gen_{attr}.txt"), 'a', encoding='utf-8')
    dirty_gen_prompt_file = open(os.path.join(enhanced_gen_directory, f"prompt_ans_dirty_gen_{attr}.txt"), 'a', encoding='utf-8')
    clean_gen_file = open(os.path.join(enhanced_gen_directory, f"clean_gen_{attr}.txt"), 'a', encoding='utf-8')
    dirty_gen_file = open(os.path.join(enhanced_gen_directory, f"dirty_gen_{attr}.txt"), 'a', encoding='utf-8')
    # 每次调用写入分隔符
    clean_gen_prompt_file.write(sep)
    dirty_gen_prompt_file.write(sep)
    clean_gen_file.write(sep)
    dirty_gen_file.write(sep)
    wrong_values = []
    right_values = []
    actual_right_values = []
    for idx, _, label in current_index_value_label_dict[attr]:
        if label == 1:
            # 与clean_csv对比，确定第一个值即dirty_csv.loc[int(idx), attr]是错误的还是正确的
            if dirty_csv.loc[int(idx), attr] != clean_csv.loc[int(idx), attr]:
                # 如果实际是错误的，则attr的脏值合并上related_attrs的干净值加入wrong_values
                wrong_values.append({attr: dirty_csv.loc[int(idx), attr], **clean_csv.loc[int(idx), related_attrs].to_dict()})
                right_values.append(clean_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            else:
                # 如果实际是正确的，则attr的干净值合并上related_attrs的干净值加入right_values
                actual_right_values.append(clean_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
                # 使用更安全的方法删除和添加元素
                # 找到要删除的元素的索引
                for i, (stored_idx, stored_value, stored_label) in enumerate(index_value_label_dict[attr]):
                    if stored_idx == idx and stored_label == label:
                        # 找到匹配的元素，现在删除它
                        del index_value_label_dict[attr][i]
                        # 添加新元素
                        # 添加新元素，使用正确的字典格式
                        index_value_label_dict[attr].append((idx, clean_csv.loc[int(idx), [attr] + related_attrs].to_dict(), 0))
                        break  # 找到后立即退出循环
                
    if len(right_values) == 0 and len(wrong_values) == 0 and len(actual_right_values) == 0:
        logger.warning(f"No data available for attribute {attr} to generate enhanced data.")
        return
            
            
    max_vals = 20
    if len(wrong_values) > max_vals:
        wrong_values_tmp = wrong_values[:max_vals]
    else:
        wrong_values_tmp = wrong_values
    if len(right_values) > max_vals:
        right_values_tmp = right_values[:max_vals]
    else:
        right_values_tmp = right_values
    if len(actual_right_values) > max_vals:
        actual_right_values_tmp = actual_right_values[:max_vals]
    else:
        actual_right_values_tmp = (actual_right_values + right_values_tmp)[:max_vals]
    
    # 创建变量来保存这次运行生成的增强数据
    new_clean_data = []
    new_dirty_data = []
    
    # 处理干净数据生成
    clean_gen_ans = ""
    if num_gen > 15:
        # 分批生成，每批20个
        num_batches = (num_gen + 14) // 15  # 向上取整
        for batch in range(num_batches):
            batch_size = min(15, num_gen - batch * 15)
            clean_gen_prompt = create_clean_gen_inst_prompt(actual_right_values_tmp, attr, num_gen=batch_size)
            batch_clean_ans = get_ans_from_llm(clean_gen_prompt, api_use=API_USE)
            clean_gen_ans += batch_clean_ans + "\n\n"
            clean_gen_prompt_file.write('*'*20 + f' batch {batch+1} prompt ' + '*'*20 + '\n' + clean_gen_prompt + '\n' + '*'*20 + f' batch {batch+1} answer ' + '*'*20 + '\n' + batch_clean_ans + '\n\n\n\n\n\n')
    else:
        # 一次性生成
        clean_gen_prompt = create_clean_gen_inst_prompt(actual_right_values_tmp, attr, num_gen=num_gen)
        clean_gen_ans = get_ans_from_llm(clean_gen_prompt, api_use=API_USE)
        clean_gen_prompt_file.write('*'*20 + ' prompt ' + '*'*20 + '\n' + clean_gen_prompt + '\n' + '*'*20 + ' answer ' + '*'*20 + '\n' + clean_gen_ans + '\n\n\n\n\n\n')
    clean_gen_prompt_file.close()
    clean_gen_file.write(clean_gen_ans)
    clean_gen_file.close()
    clean_info = extract_enhanced_info(clean_gen_ans, attr)

    if (len(wrong_values_tmp) ==0):
        logger.warning(f"No wrong values available for attribute {attr} to generate dirty data.")
        return
    # 处理干净数据，获取filtered_clean
    filtered_clean = []
    # filtered_clean.extend(right_values+actual_right_values)
    for clean in clean_info:
        if len(clean) < 4 or len(clean[3]) == 0 or not isinstance(clean[3], dict) or len(clean[3].keys()) < len([attr]+related_attrs) :
            continue

        try:
            if clean[0] in all_attrs and str(clean[-1]).strip() not in right_values and str(
                    clean[-1]).strip() not in wrong_values:
                enhanced_gen_dict[attr]['clean'].append(clean[3])
                filtered_clean.append(clean[3])
                # 添加到新数据列表
                new_clean_data.append(clean[3])
        except IndexError as e:
            logger.error(f"\nError: {e}\n Handling Value: {clean}\n Processing attribute: {attr}\n")
    
    # 处理脏数据生成（修改为对filtered_clean中的数据注入错误）
    dirty_gen_ans = ""
    num_dirty_to_generate = len(filtered_clean)  # 使用filtered_clean的长度作为需要生成的脏数据数量
    
    # 对filtered_clean分批，每批15个，注入错误
    if num_dirty_to_generate > 15:
        # 分批生成，每批15个
        num_batches = (num_dirty_to_generate + 14) // 15  # 向上取整
        for batch in range(num_batches):
            start_idx = batch * 15
            end_idx = min((batch + 1) * 15, num_dirty_to_generate)
            clean_for_error_injection = filtered_clean[start_idx:end_idx]
            
            dirty_gen_prompt = create_dirty_gen_inst_prompt(clean_for_error_injection, right_values_tmp, wrong_values_tmp, attr, num_errors=len(clean_for_error_injection))
            batch_dirty_ans = get_ans_from_llm(dirty_gen_prompt, api_use=API_USE)
            dirty_gen_ans += batch_dirty_ans + "\n\n"
            dirty_gen_prompt_file.write('*'*20 + f' batch {batch+1} prompt ' + '*'*20 + '\n' + dirty_gen_prompt + '\n' + '*'*20 + f' batch {batch+1} answer ' + '*'*20 + '\n' + batch_dirty_ans + '\n\n\n\n\n\n')
    else:
        # 一次性生成
        dirty_gen_prompt = create_dirty_gen_inst_prompt(filtered_clean, right_values_tmp, wrong_values_tmp, attr, num_errors=num_dirty_to_generate)
        dirty_gen_ans = get_ans_from_llm(dirty_gen_prompt, api_use=API_USE)
        dirty_gen_prompt_file.write('*'*20 + ' prompt ' + '*'*20 + '\n' + dirty_gen_prompt + '\n' + '*'*20 + ' answer ' + '*'*20 + '\n' + dirty_gen_ans + '\n\n\n\n\n\n')
    
    dirty_gen_prompt_file.close()
    dirty_gen_file.write(dirty_gen_ans)
    dirty_gen_file.close()
    dirty_info = extract_enhanced_info(dirty_gen_ans, attr)
    filtered_dirty = []
    filtered_dirty.extend(wrong_values)
    for dirty in dirty_info:
        try:
            if len(dirty) < 4 or len(dirty[3]) == 0 or len(dirty[3].keys()) < len([attr]+related_attrs) or not isinstance(dirty[3], dict):
                continue
            if dirty[0] in all_attrs and str(dirty[-1]).strip() not in right_values and str(
                    dirty[-1]).strip() not in wrong_values:
                enhanced_gen_dict[attr]['dirty'].append(dirty[3])
                # 将字典格式的数据添加到filtered_dirty中，保持数据类型一致
                filtered_dirty.append(dirty[3])
                # 添加到新数据列表
                new_dirty_data.append(dirty[3])
        except IndexError as e:
            logger.error(f"\nError: {e}\n Handling Value: {dirty}\n Processing attribute: {attr}\n")
    
    # 只保存这次运行新生成的数据，而不是整个enhanced_gen_dict
    if new_clean_data:
        clean_gen_res_file = open(os.path.join(enhanced_gen_directory, f"clean_gen_res_{attr}.txt"), 'a', encoding='utf-8')
        for clean_dict in new_clean_data:
            json.dump(clean_dict, clean_gen_res_file)
            clean_gen_res_file.write('\n')
        clean_gen_res_file.close()
    
    if new_dirty_data:
        dirty_gen_res_file = open(os.path.join(enhanced_gen_directory, f"dirty_gen_res_{attr}.txt"), 'a', encoding='utf-8')
        for dirty_dict in new_dirty_data:
            json.dump(dirty_dict, dirty_gen_res_file)
            dirty_gen_res_file.write('\n')
        dirty_gen_res_file.close()

def gen_err_funcs(attr, err_gen_dict):  
    related_attrs = list(related_attrs_dict[attr])  
    err_gen_prompt_file = open(os.path.join(funcs_directory, f"prompt_ans_error_gen_{attr}.txt"), 'a', encoding='utf-8')
    err_gen_file = open(os.path.join(funcs_directory, f"error_gen_{attr}.txt"), 'w', encoding='utf-8')
    wrong_values = []
    right_values = []
    used_idx_list = {}
    for idx, _, label in index_value_label_dict[attr]:
        if label == 1:  # wrong 
            wrong_values.append(dirty_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            used_idx_list[idx] = 1
        elif label == 0:  # right
            right_values.append(dirty_csv.loc[int(idx), [attr] + related_attrs ].to_dict())
            used_idx_list[idx] = 1
    filtered_error = [str(vals) for vals in wrong_values]
    if len(filtered_error) == 0:
        return False
    max_err_num = 20
    if max_err_num > (int(len(filtered_error)/2)+1):
        max_err_num = int(len(filtered_error)/2)+1
    filtered_error_sublists = split_list_to_sublists(filtered_error, max_err_num)
    if len(filtered_error_sublists) > 2:
       filtered_error_sublists = filtered_error_sublists[:2]
    funcs_for_attr = {}
    max_err_num = min(max_err_num, len(right_values))
    with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as a_executor:
        a_results = [a_executor.submit(subtask_func_gen, attr, filtered_error_sublists[temp_idx], temp_idx, random.sample(right_values, max_err_num)) for temp_idx in range(len(filtered_error_sublists))]
        for a_future in as_completed(a_results):  
            attr_name, funcs_for_attr_gen = a_future.result()
            funcs_for_attr.update(funcs_for_attr_gen)
    func_extract_file = open(os.path.join(funcs_directory, f"funcs_zgen_{attr}.txt"), 'w', encoding='utf-8')
    temp_clean_flist_str = "\n".join(funcs_for_attr[attr]['clean'])
    func_extract_file.write(temp_clean_flist_str)
    func_extract_file.close()
    return funcs_for_attr


def execute_func(function_code, val, attr):
    # Define a local scope to execute our function
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
        return -1  # Returning -1 to indicate failure
    return 1 if result else 0  # Returning 1 for True, 0 for False


def task_guide_gen(attr_name, uni_vals, distri_analy_content, prompt_content, guide_content):
    attr_analy_content = distri_analy_content[attr_name]
    prompt = guide_gen_prompt(attr_name, dataset, uni_vals, dirty_csv, attr_analy_content)
    while True:
        try:
            res_content = get_ans_from_llm(prompt, api_use=API_USE)
            break
        except Exception as eee:
            print(eee, f'while guide_gen {attr_name}')
    prompt_content[attr_name] = prompt
    guide_content[attr_name] = res_content
    with open(os.path.join(guide_directory, f'prompt_{attr_name}.txt'), 'w', encoding='utf-8') as file:
        file.write(prompt)
    with open(os.path.join(guide_directory, f'guide_{attr_name}.txt'), 'w', encoding='utf-8') as file:
        file.write(res_content)


def task_func_gen(attr_name, err_gen_dict):
    funcs_for_attr = gen_err_funcs(attr_name, err_gen_dict)
    if funcs_for_attr:
        para_file.write(f"{attr_name} func_num:{len(funcs_for_attr[attr_name]['clean'])}\n")
        return funcs_for_attr
    else:
        return {attr_name: {'clean': [], 'dirty': []}}


def task_det_initial(attr_name, error_checking_res_directory, indices):
    """
    处理特定属性的初始错误检测，针对指定的indices
    
    Args:
        attr_name: 属性名
        error_checking_res_directory: 错误检查结果目录
        indices: 数据集中的索引列表
    """
    # 清空文件，以追加模式写入
    if not os.path.exists(os.path.join(error_checking_res_directory, f'error_checking_{attr_name}.txt')):
        error_checking_file = open(os.path.join(error_checking_res_directory, f'error_checking_{attr_name}.txt'), 'w', encoding='utf-8')
        error_checking_file.close()
    
    related_attrs = list(related_attrs_dict[attr_name])
    
    # 为每个索引创建数据字典
    df_indices = ["{" + ",".join(f'"{col}":"{dirty_csv.loc[idx, col]}"' for col in [attr_name] + related_attrs) + "}" for idx in indices]
    
    # 将数据分成子列表进行处理
    split_values = split_list_to_sublists(df_indices, err_check_val_num_per_query)
    
    # 将索引也分成对应的子列表
    split_indices = split_list_to_sublists(indices, err_check_val_num_per_query)
    
    error_response = ''
    # 改为单线程处理，便于调试
    for sub_list_values, sub_list_indices in zip(split_values, split_indices):
        try:
            error_response += subtask_det_initial(sub_list_values, attr_name, sub_list_indices) + '\n'
        except Exception as e:
            print(f"处理属性 {attr_name} 的子任务时出错: {str(e)}")
            import traceback
            traceback.print_exc()


def normalize_string(s):
    return str(s.replace(" \\", "\\")
               .replace("\\\\", "\\")
               .replace("\\", "")
               .replace(", ", ",")
               .replace(": ", ":")
               .replace("'", '"'))


def process_attr_train_feat(attr, dirty_csv, det_right_list, det_wrong_list, related_attrs_dict, enhanced_gen_dict, funcs_for_attr, feature_all_dict, resp_path):
    fasttext_model = fasttext.load_model('./cc.en.300.bin')
    fasttext_dimension = len(dirty_csv.columns)
    fasttext.util.reduce_model(fasttext_model, fasttext_dimension)
    feature_list, label_list = prep_train_feat(attr, dirty_csv, det_right_list, det_wrong_list, related_attrs_dict, enhanced_gen_dict, funcs_for_attr, fasttext_model, feature_all_dict, resp_path)
    return attr, feature_list, label_list


def task_distri_analys(attr, analyzer, dist_dir):
    output_file = os.path.join(dist_dir, f'ori_distri_analys_{attr}.txt')
    distr_prompt_file = os.path.join(dist_dir, f'prompt_distri_analys_{attr}.txt')
    llm_prompt, examples = analyzer.generate_llm_prompt(attr)
    llm_response = get_ans_from_llm(llm_prompt, api_use=API_USE)
    analyze_content = analyzer.analyze_data(attr, llm_response, output_file)
    with open(distr_prompt_file, 'w', encoding='utf-8') as f:
        f.write(llm_prompt)
    with open(os.path.join(dist_dir, f'distri_analys_{attr}.txt'), 'w', encoding='utf-8') as f:
        f.write(analyze_content)
    return attr, analyze_content


def single_val_feat(val, fasttext_m, funcs_for_attr, attr, idx, all_attrs, feature_all_dict, resp_path):
    # feature = [handle_func_exec(func, val, attr) for func in funcs_for_attr[attr]['clean']]
    feature = [handle_func_exec(func, val, attr) if handle_func_exec(func, val, attr) != -1 else 0
    for func in funcs_for_attr[attr]['clean']
    ]
    if idx == -1:
        for a_val in val.values():
            feature.extend(fasttext_m.get_word_vector(str(a_val)))
        return feature
    else:
        if feature_all_dict is not None:
            fasttext_feat = feature_all_dict[(idx, all_attrs.index(attr))].get('fasttext_feat', [])
            if len(fasttext_feat) == 0 or len(fasttext_feat) < len(all_attrs):
                fasttext_feat = []
                fasttext_m = fasttext.load_model('./cc.en.300.bin')
                fasttext_dimension = len(all_attrs)
                fasttext.util.reduce_model(fasttext_m, fasttext_dimension)
                # 检查val的类型，如果是numpy.ndarray，直接遍历；如果是字典或Series，使用values()
                if isinstance(val, np.ndarray):
                    for a_val in val:
                        fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
                else:
                    # 假设是字典或Pandas Series
                    try:
                        for a_val in val.values():
                            fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
                    except AttributeError:
                        # 如果还是没有values()方法，直接遍历
                        for a_val in val:
                            fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
            feature.extend(fasttext_feat)
        else:
            fasttext_m = fasttext.load_model('./cc.en.300.bin')
            fasttext_dimension = len(all_attrs)
            fasttext.util.reduce_model(fasttext_m, fasttext_dimension)
            fasttext_feat = []
            # 检查val的类型，如果是numpy.ndarray，直接遍历；如果是字典或Series，使用values()
            if isinstance(val, np.ndarray):
                for a_val in val:
                    fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
            else:
                # 假设是字典或Pandas Series
                try:
                    for a_val in val.values():
                        fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
                except AttributeError:
                    # 如果还是没有values()方法，直接遍历
                    for a_val in val:
                        fasttext_feat.extend(fasttext_m.get_word_vector(str(a_val)))
            feature.extend(fasttext_feat)
        return idx, feature


def prep_train_feat(attr, dirty_csv, det_right_list, det_wrong_list, related_attrs_dict, enhanced_gen_dict, funcs_for_attr, fasttext_m, feature_all_dict, resp_path):
    feature_list = []
    label_list = []
    related_attrs = list(related_attrs_dict[attr])
    right_values = [(idx, dirty_csv.loc[idx, [attr]+related_attrs].to_dict()) for idx, a in det_right_list if a == attr]
    wrong_values = [(idx, dirty_csv.loc[idx, [attr]+related_attrs].to_dict()) for idx, a in det_wrong_list if a == attr]
    
    for idx, val in tqdm(right_values, ncols=120, desc=f"Processing {attr} right values"):
        feature = single_val_feat(val, fasttext_m, funcs_for_attr, attr, -1, list(dirty_csv.columns), feature_all_dict, resp_path)
        feature_list.append(feature)
        label_list.append(0)
    for idx, val in tqdm(wrong_values, ncols=120, desc=f"Processing {attr} wrong values"):
        feature = single_val_feat(val, fasttext_m, funcs_for_attr, attr, -1, list(dirty_csv.columns), feature_all_dict, resp_path)
        feature_list.append(feature)
        label_list.append(1)
    for val in tqdm(enhanced_gen_dict[attr]['clean'], ncols=120, desc=f"Processing {attr} generated clean data"):
        if len(enhanced_gen_dict[attr]['clean']) == 0 or len(val) < len([attr]+related_attrs) or not isinstance(val, dict):
            continue

        feature = single_val_feat(val, fasttext_m, funcs_for_attr, attr, -1, list(dirty_csv.columns), feature_all_dict, resp_path)
        feature_list.append(feature)
        label_list.append(0)
    for val in tqdm(enhanced_gen_dict[attr]['dirty'], ncols=120, desc=f"Processing {attr} generated dirty data"):
        if len(enhanced_gen_dict[attr]['dirty']) == 0 or len(val) < len([attr]+related_attrs) or not isinstance(val, dict):
            continue
        
        feature = single_val_feat(val, fasttext_m, funcs_for_attr, attr, -1, list(dirty_csv.columns), feature_all_dict, resp_path)
        feature_list.append(feature)
        label_list.append(1)
    return feature_list, label_list


def make_predictions(col, attr, dirty_csv, model_col, related_attrs_dict, funcs_for_attr, feature_all_dict, resp_path):
    if attr not in model_col.keys():
        return []    
    model = model_col[attr]
    test_feat_list = []
    related_attrs = list(related_attrs_dict[attr])
    columns = list(dirty_csv.columns)
    
    # 改为单线程处理
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
    if feature_list is None:
        return attr, None, 'mlp', 'optimizer', "None", 500
    elif len(feature_list) == 0 or len(feature_list[0]) == 0:
        return attr, None, 'mlp', 'optimizer', "None", 500
    
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
        verbose=True                 
    )
    
    model.fit(feat_np, label_np)
    return attr, model, 'mlp', 'optimizer', model, num_epochs


def process_related_attr(RELATED_ATTRS, RELATED_ATTRS_READ, REL_TOP, read_path, resp_path, clean_csv, dirty_csv, all_attrs):
    related_attrs_dict = {}
    gt_wrong_dict = {}
    if RELATED_ATTRS and RELATED_ATTRS_READ:
        with open(os.path.join(read_path, 'related_attrs_dict.json'), 'r', encoding='utf-8') as f:
            related_attrs_dict = json.load(f)
        copy_file(read_path, resp_path, 'related_attrs_dict.json')
    elif RELATED_ATTRS and not RELATED_ATTRS_READ:
        nmi_results = cal_all_column_nmi(dirty_csv)
        related_attrs_dict = cal_strong_res_column_nmi(nmi_results, rel_top=REL_TOP)
        with open(os.path.join(resp_path, 'related_attrs_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(related_attrs_dict, f, ensure_ascii=False, indent=4)
    elif not RELATED_ATTRS:
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


def process_cluster(n_method, CLUSTER_READ, dataset, read_path, resp_path, dirty_csv, all_attrs, related_attrs_dict, pre_funcs_for_attr):
    cluster_index_dict = {}  
    center_value_dict = {}  
    feature_all_dict = defaultdict(default_dict_of_lists)
    if not CLUSTER_READ:
        for col in range(len(all_attrs)):
            try:
                col_result, center_list, cluster_list, val_feat_dict, feature_dict_attr = cluster(dataset, 'KMeans', n_method, col, related_attrs_dict, pre_funcs_for_attr, resp_path)
                cluster_list.insert(0, center_list)
                cluster_index_dict[all_attrs[col]] = cluster_list
                feature_all_dict.update(feature_dict_attr)
            except Exception as e:
                print(f"列 {col} ({all_attrs[col]}) 处理出错: {str(e)}")
                import traceback
                traceback.print_exc()
                raise  # 重新抛出异常，停止程序
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
    elif CLUSTER_READ:
        with open(os.path.join(read_path, 'center_value_dict.json'), 'r', encoding='utf-8') as src_file:
            center_value_dict = json.load(src_file)
        with open(os.path.join(resp_path, 'center_value_dict.json'), 'w', encoding='utf-8') as dst_file:
            json.dump(center_value_dict, dst_file, ensure_ascii=False, indent=4)
                    
        with open(os.path.join(read_path, 'cluster_index_dict.json'), 'r', encoding='utf-8') as f:
            cluster_index_dict = json.load(f)
        with open(os.path.join(resp_path, 'cluster_index_dict.json'), 'w', encoding='utf-8') as dst_file:
           json.dump(cluster_index_dict, dst_file, ensure_ascii=False, indent=4)
        cluster_index_dict = {
                    attr: [[int(idx) for idx in cluster] for cluster in clusters]
                    for attr, clusters in cluster_index_dict.items()
                }
        if os.path.exists(os.path.join(read_path, 'cluster_feat_dict.pkl')):
            copy_file(read_path, resp_path, 'cluster_feat_dict.pkl')
            with open(os.path.join(read_path, 'cluster_feat_dict.pkl'), 'rb') as f:
                feature_all_dict = pickle.load(f)
    return cluster_index_dict, center_value_dict, feature_all_dict


def process_distri_analys(DISTRI_ANALYSIS, DISTRI_ANALYSIS_READ, read_path, resp_path, dirty_csv, all_attrs):
    dist_dir = os.path.join(resp_path, 'distri_analys')
    os.makedirs(dist_dir, exist_ok=True)
    distri_analy_content = {}
    if DISTRI_ANALYSIS and DISTRI_ANALYSIS_READ:
        distri_analy_read_dir = os.path.join(read_path, 'distri_analys')
        copy_read_files_in_dir(dist_dir, distri_analy_read_dir)
        for attr in all_attrs:
            dist_dir_file = os.path.join(dist_dir, f'distri_analys_{attr}.txt')
            with open(dist_dir_file, 'r', encoding='utf-8') as file:
                distri_analy_content[attr] = file.read()
    elif DISTRI_ANALYSIS and not DISTRI_ANALYSIS_READ:
        analyzer = LLMDataDistrAnalyzer(dirty_csv)
        with multiprocessing.Pool(len(all_attrs)) as pool:
            results = [pool.apply_async(task_distri_analys, args=(attr, analyzer, dist_dir)) for attr in all_attrs]
            for result in results:
                attr, content = result.get()
                distri_analy_content[attr] = content
    else:
        for attr in all_attrs:
            distri_analy_content[attr] = ''

    return distri_analy_content


def process_guidlines(GUIDE_USE, GUIDE_READ, dataset, read_path, read_guide_path, resp_path, dirty_csv, all_attrs, guide_directory, cluster_index_dict, distri_analy_content):
    guide_content = {}
    prompt_content = {}
    if GUIDE_USE and GUIDE_READ:
        copy_read_files_in_dir(guide_directory, read_guide_path)
        for attr in all_attrs:
            attr_analy_content = distri_analy_content[attr]
            file_path = os.path.join(read_guide_path, f'{dataset}_{attr}_ref_knowledge.txt')
            prompt = guide_gen_prompt(attr, dataset, cluster_index_dict[attr][0], dirty_csv, attr_analy_content)
            prompt_content[attr] = prompt
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    guide_content[attr] = file.read()
            elif os.path.exists(os.path.join(read_path, f'guide/guide_{attr}.txt')):
                with open(os.path.join(read_path, f'guide/guide_{attr}.txt'), 'r', encoding='utf-8') as file:
                    guide_content[attr] = file.read()
            else:
                continue
    elif GUIDE_USE:
        guide_content = {}
        prompt_content = {}
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(task_guide_gen, attr, cluster_index_dict[attr][0], distri_analy_content, prompt_content, guide_content) for attr in all_attrs]
            for future in as_completed(results):  
                result = future.result()
    return guide_content

def calculate_jsd(p, q):
    """
    计算Jensen-Shannon Divergence (JSD)
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    其中 M = 0.5 * (P + Q)
    """
    # 确保p和q是概率分布（和为1）
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # 归一化
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # 避免零值
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)
    
    # 计算M
    m = 0.5 * (p + q)
    
    # 计算KL散度
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    
    # 计算JSD
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


def calculate_ksd(sample1, sample2):
    """
    计算Kolmogorov-Smirnov Distance (KSD)
    使用scipy的ks_2samp函数计算两个样本之间的KS距离
    """
    # 确保输入是numpy数组
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    
    # 计算KS统计量和p值
    ks_statistic, p_value = stats.ks_2samp(sample1, sample2)
    
    # 返回KS统计量作为距离度量
    return ks_statistic


def process_select_optimal_cluster(
    enhanced_gen_dict, cluster_index_dict, dirty_csv, all_attrs, related_attrs_dict,
     pre_funcs_for_attr, resp_path, logger, index_value_label_dict, residual_method='both',
     previously_selected_clusters=None
):
    """
    从聚类结果中选出一个聚类并与增强数据合并，使其与dirty_csv的残差最小
    
    Args:
        enhanced_gen_dict: 增强数据字典
        cluster_index_dict: 聚类索引字典
        dirty_csv: 脏数据DataFrame
        all_attrs: 所有属性列表
        related_attrs_dict: 相关属性字典
        pre_funcs_for_attr: 预处理函数字典
        resp_path: 响应路径
        logger: 日志记录器
        index_value_label_dict: 索引值标签字典
        residual_method: 残差计算方法，'jsd'、'ksd'或'both'
        previously_selected_clusters: 之前选择过的聚类，格式为 {attr: [cluster_idx1, cluster_idx2, ...]}
    
    Returns:
        optimal_cluster_info_dict: 最优聚类信息字典
    """
    optimal_cluster_info_dict = {}
    
    # 初始化之前选择过的聚类字典
    if previously_selected_clusters is None:
        previously_selected_clusters = {}
    
    logger.info("开始选择最优聚类...")
    
    # 全局加载FastText模型，避免重复加载
    try:
        fasttext_model = fasttext.load_model('./cc.en.300.bin')
        # fasttext_dimension = len(dirty_csv.columns)
        fasttext_dimension = len(related_attrs_dict[next(iter(related_attrs_dict))]) + 1
        fasttext.util.reduce_model(fasttext_model, fasttext_dimension)
    except Exception as e:
        logger.error(f"加载FastText模型失败: {str(e)}")
        return optimal_cluster_info_dict

    global_cache = feat_gen_global_cache(dirty_csv, related_attrs_dict)

    # 并行处理每个属性
    def process_attr(attr):
        min_residual = float('inf')
        attr_optimal_cluster = None
        
        logger.info(f"处理属性: {attr}")
        
        # 获取该属性的聚类信息
        if attr not in cluster_index_dict:
            logger.warning(f"属性 {attr} 不在聚类索引字典中，跳过")
            return None
            
        clusters = cluster_index_dict[attr]
        if len(clusters) == 0:
            logger.warning(f"属性 {attr} 没有有效聚类，跳过")
            return None
            
        related_attrs = list(related_attrs_dict[attr])
        col_num = list(dirty_csv.columns).index(attr)

        # 从feature_all_dict中提取ref_features，不再重复计算
        ref_data = dirty_csv.loc[:, [attr] + related_attrs]
        ref_df = pd.DataFrame(ref_data) if not ref_data.empty else pd.DataFrame()
        col_num = list(ref_df.columns).index(attr)
        # ref_features, ref_feature_dict = feat_gen_df(ref_df, col_num, attr, pre_funcs_for_attr, resp_path)
        ref_features, _, scaler = feat_gen_df_incremental(ref_df, col_num, attr, pre_funcs_for_attr, resp_path, global_cache)
        ref_features = np.array(ref_features, dtype=np.float64)
        # 处理可能的NaN或无限值
        ref_features = np.nan_to_num(ref_features)

        # 获取该属性之前选择过的聚类
        attr_previously_selected = previously_selected_clusters.get(attr, [])
        
        # 遍历聚类
        for cluster_idx, cluster_indices in enumerate(clusters[1:], start=0):
            # 跳过已经选择过的聚类
            if cluster_idx in attr_previously_selected:
                logger.info(f"跳过属性 {attr} 的聚类 {cluster_idx}，因为之前已经选择过")
                continue
                
            if len(cluster_indices) == 0:
                continue
                
            cluster_data = dirty_csv.loc[cluster_indices, [attr] + related_attrs]
            
            # 合并增强数据
            enhanced_data = []
            if attr in enhanced_gen_dict:
                enhanced_data.extend(enhanced_gen_dict[attr].get('clean', []))
                enhanced_data.extend(enhanced_gen_dict[attr].get('dirty', []))
            enhanced_df = pd.DataFrame(enhanced_data) if enhanced_data else pd.DataFrame()
            
            # 合并标注数据
            labeled_data = []
            if attr in index_value_label_dict:
                for idx, value, label in index_value_label_dict[attr]:
                    # 只添加不在当前聚类中的标注数据，避免重复
                    if idx not in cluster_indices:
                        labeled_data.append(value)
            labeled_df = pd.DataFrame(labeled_data) if labeled_data else pd.DataFrame()
            
            # 合并所有数据
            combined_data = pd.concat([cluster_data, enhanced_df, labeled_df], ignore_index=True)
            if combined_data.empty:
                continue

            # === 计算 combined_data 的特征 ===
            # col_num = list(combined_data.columns).index(attr)
            # combined_feature_list, combined_feature_dict = feat_gen_df(combined_data, col_num, attr, pre_funcs_for_attr, resp_path)
            # combined_feature_list = np.array(combined_feature_list, dtype=np.float64)
            combined_feature_list, _, _ = feat_gen_df_incremental(
                combined_data, col_num, attr, pre_funcs_for_attr, resp_path, global_cache, scaler
            )
            # 处理可能的NaN或无限值
            combined_feature_list = np.nan_to_num(combined_feature_list)
            # === 计算残差 ===
            try:
                if residual_method in ['jsd', 'both']:
                    # 使用直方图估计分布
                    hist_comb, _ = np.histogram(combined_feature_list.flatten(), bins=30, density=True)
                    hist_ref, _ = np.histogram(ref_features.flatten(), bins=30, density=True)
                    jsd_residual = calculate_jsd(hist_comb, hist_ref)
                else:
                    jsd_residual = float('inf')

                if residual_method in ['ksd', 'both']:
                    # 对特征分量取平均后计算KSD
                    # 检查数组维度，如果是1维则直接使用，否则在axis=1上取平均
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

                # === 选择最优聚类 ===
                if combined_residual < min_residual:
                    min_residual = combined_residual
                    attr_optimal_cluster = {
                        'cluster_idx': cluster_idx,
                        'cluster_indices': cluster_indices,
                        'jsd_residual': jsd_residual,
                        'ksd_residual': ksd_residual,
                        'combined_residual': combined_residual,
                        'enhanced_data': enhanced_data
                    }

            except Exception as e:
                logger.error(f"计算属性 {attr} 聚类 {cluster_idx} 残差时出错: {str(e)}")
                continue

        return attr, attr_optimal_cluster

    # 改为单线程处理，避免多线程导致的性能问题和卡死
    for attr in all_attrs:
        result = process_attr(attr)
        if result:
            attr, attr_optimal_cluster = result
            optimal_cluster_info_dict[attr] = attr_optimal_cluster

    if optimal_cluster_info_dict:
        logger.info("最优聚类信息:")
        for attr, cluster_info in optimal_cluster_info_dict.items():
            logger.info(f"属性 {attr} 聚类 {cluster_info['cluster_idx']}，综合残差 {cluster_info['combined_residual']:.4f}")
    else:
        logger.warning("未找到有效的最优聚类")
    
    return optimal_cluster_info_dict

def process_error_checking(ERROR_CHECKING_READ, read_error_checking_path, all_attrs, error_checking_res_directory):
    if ERROR_CHECKING_READ:
        copy_read_files_in_dir(error_checking_res_directory, read_error_checking_path)

    else:
        # 改为单线程处理，便于调试
        for attr in all_attrs:
            try:
                task_det_initial(attr, error_checking_res_directory)
            except Exception as e:
                print(f"处理属性 {attr} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()


def measure_llm_label(resp_path, clean_csv, all_attrs, related_attrs_dict, gt_wrong_dict, index_value_label_dict):
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
        for idx, llm_lstr, llm_label in index_value_label_dict[attr]:
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
    llm_label_eval_file.write(f"Overall Right data labeling accuracy: {1-overall_miss_wrong_num/(overall_lright_num)+1e-6} ({overall_lright_num-overall_miss_wrong_num}/{overall_lright_num})\n\n")
    llm_label_eval_file.close()
    return 'Done'


def err_pat_in_text_attr(attr):
    pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"[^"]*",\s*\n\s*"has_error_in_{attr}_value":\s*true'
    return pattern


def right_pat_in_text_attr(attr):
    pattern = fr'"value_row":\s*(".*?"),\s*\n\s*"error_analysis":\s*"[^"]*",\s*\n\s*"has_error_in_{attr}_value":\s*false'
    return pattern


def extract_llm_label_res(all_attrs, error_checking_res_directory, indices_dict=None):
    """
    从LLM标注结果中提取标签，针对指定的indices
    
    Args:
        all_attrs: 所有属性列表
        error_checking_res_directory: 错误检查结果目录
        indices_dict: 指定要提取的indices字典，格式为 {attr: [idx1, idx2, ...]}，如果为None则提取所有标签
    
    Returns:
        索引值标签字典
    """
    # 如果indices_dict为None，则提取所有标签
    if indices_dict is None:
        return extract_all_llm_label_res(all_attrs, error_checking_res_directory)
    all_extracted_values = defaultdict(list)
    index_value_label_dict = defaultdict(list)
    
    for attr in all_attrs:
        content = ""
        with open(os.path.join(error_checking_res_directory, f'error_checking_{attr}.txt'), 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('\\+', '').replace('\\n', '\n')
        
        # 提取错误值
        wrong_pattern = err_pat_in_text_attr(attr)
        matches = re.finditer(wrong_pattern, content)
        all_extracted_values[attr].extend([match.group(1).replace("':'", "': '").replace(',', ', ').replace(',  ', ', ').replace('"', "'") for match in matches])
        all_extracted_values[attr] = [normalize_string(match).replace('"{', '{', 1)[:-1] for match in all_extracted_values[attr]]
        all_extracted_values[attr] = list(set(all_extracted_values[attr]))
        
        # 处理冲突值
        right_pattern = right_pat_in_text_attr(attr)
        right_matches = re.finditer(right_pattern, content)
        right_matches = [match.group(1).replace("':'", "': '").replace(',', ', ').replace(',  ', ', ').replace('"', "'") for match in right_matches]
        right_matches = [normalize_string(match).replace('"{', '{', 1)[:-1] for match in right_matches]
        all_extracted_values[attr] = [extr_vals for extr_vals in all_extracted_values[attr] if extr_vals not in right_matches]
    
    # 使用指定的indices
    for attr in all_attrs:
        if attr in indices_dict:
            indices = indices_dict[attr]
            temp_list = []
            for idx in indices:
                # 获取该索引处的值
                related_attrs = list(related_attrs_dict[attr])
                value = dirty_csv.loc[idx, [attr] + related_attrs].to_dict()
                # 检查该值是否在错误值列表中
                if normalize_string(str(value)) in all_extracted_values[attr]:
                    temp_list.append((idx, value, 1))
                else:
                    temp_list.append((idx, value, 0))
            index_value_label_dict[attr] = temp_list
    
    return index_value_label_dict


def extract_all_llm_label_res(all_attrs, error_checking_res_directory):
    """
    从LLM标注结果中提取所有标签，不限制indices
    
    Args:
        all_attrs: 所有属性列表
        error_checking_res_directory: 错误检查结果目录
    
    Returns:
        索引值标签字典，格式为 {attr: [(idx, value, label), ...]}
    """
    index_value_label_dict = defaultdict(list)
    
    for attr in all_attrs:
        content = ""
        with open(os.path.join(error_checking_res_directory, f'error_checking_{attr}.txt'), 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('\\+', '').replace('\\n', '\n')
        
        # 使用正则表达式匹配每个块
        block_pattern = r'// indices:\s*\[(.*?)\][\s\S]*?```json\s*(\{[\s\S]*?\})\s*```'
        blocks = re.findall(block_pattern, content, re.DOTALL)
        
        for indices_str, json_content in blocks:
            try:
                # 解析indices列表，先去除换行符
                indices_str_clean = indices_str.replace('\n', ' ').strip()
                # 直接手动分割字符串，提取数字
                indices = [int(x) for x in re.findall(r'\d+', indices_str_clean)]
                
                # 解析JSON内容
                json_data = json.loads(json_content)
                entries = json_data.get('entries', [])
                
                # 确保indices和entries的数量匹配
                if len(indices) != len(entries):
                    print(f"Warning: indices and entries count mismatch for {attr}")
                    continue
                
                # 处理每个条目
                for idx, entry in zip(indices, entries):
                    try:
                        # 获取该索引处的值
                        related_attrs = list(related_attrs_dict[attr])
                        value = dirty_csv.loc[idx, [attr] + related_attrs].to_dict()
                        
                        # 从JSON条目中获取标签
                        label_key = f"has_error_in_{attr}_value"
                        label = 1 if entry.get(label_key, False) else 0
                        
                        # 添加到结果字典
                        index_value_label_dict[attr].append((idx, value, label))
                    except Exception as e:
                        print(f"Error processing entry for {attr}, index {idx}: {e}")
                        continue
            except Exception as e:
                print(f"Error processing block for {attr}: {e}")
                continue
    
    return index_value_label_dict


def save_indices_labels(indices_dict, labels_dict, file_path):
    """
    保存任意indices的标注结果到文件
    
    Args:
        indices_dict: indices字典，格式为 {attr: [idx1, idx2, ...]}
        labels_dict: 标签字典，格式为 {attr: [(idx, value, label), ...]}
        file_path: 保存文件路径
    """
    save_data = {
        'indices_dict': indices_dict,
        'labels_dict': labels_dict
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)


def load_indices_labels(file_path):
    """
    从文件加载任意indices的标注结果
    
    Args:
        file_path: 加载文件路径
    
    Returns:
        indices_dict: indices字典，格式为 {attr: [idx1, idx2, ...]}
        labels_dict: 标签字典，格式为 {attr: [(idx, value, label), ...]}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        save_data = json.load(f)
    
    return save_data['indices_dict'], save_data['labels_dict']

def compare_llm_and_classifier_labels(current_index_value_label_dict, det_wrong_list_res):
    """
    比较LLM标注结果和分类器标注结果，复制current_index_value_label_dict中和模型标注不一致的数据
    
    Args:
        current_index_value_label_dict: LLM标注结果，格式为 {attr: [(idx, value, label), ...]}
        det_wrong_list_res: 分类器标注结果，格式为 [(idx, attr), ...]
    
    Returns:
        inconsistent_index_value_label_dict: 与模型标注不一致的数据，格式与current_index_value_label_dict相同
    """
    inconsistent_index_value_label_dict = defaultdict(list)
    
    # 将分类器标注结果转换为字典格式，方便查找
    classifier_labels = {}
    for idx, attr in det_wrong_list_res:
        classifier_labels[(idx, attr)] = 1  # 1表示分类器认为这是错误的
    
    # 遍历LLM标注结果
    for attr, label_list in current_index_value_label_dict.items():
        for idx, value, llm_label in label_list:
            # 获取分类器对该单元格的标注
            classifier_label = classifier_labels.get((idx, attr), 0)  # 默认为0（正确）
            
            # 比较标注结果，如果不一致则添加到结果中
            if llm_label != classifier_label:
                inconsistent_index_value_label_dict[attr].append((idx, value, llm_label))
    
    return inconsistent_index_value_label_dict

def label_prop(resp_path, dirty_path, clean_path, cluster_index_dict, index_value_label_dict, label_prop=True):
    """
    根据标注结果在聚类内扩散标签
    
    Args:
        resp_path: 响应路径
        dirty_path: 脏数据路径
        clean_path: 清洁数据路径
        cluster_index_dict: 聚类索引字典
        index_value_label_dict: 索引值标签字典
        label_prop: 是否进行标签扩散

    Returns:
        det_wrong_list: 检测到的错误列表
        det_right_list: 检测到的正确列表
    """
    det_wrong_list = []
    det_right_list = []
    
    # 首先添加已有标签的数据
    for attr, label_list in index_value_label_dict.items():
        for idx, value, label in label_list:
            if label == 1:
                det_wrong_list.append((idx, attr))
            elif label == 0:
                det_right_list.append((idx, attr))
    
    if not label_prop:
        return det_wrong_list, det_right_list
    # 然后在聚类内扩散标签
    for attr, clusters in cluster_index_dict.items():
        # 获取该属性的所有聚类中心
        center_indices = clusters[0]
        
        # 为每个聚类中心找到其标签
        center_labels = {}
        for idx, value, label in index_value_label_dict.get(attr, []):
            if idx in center_indices:
                center_labels[idx] = label
        
        # 对每个聚类中心，在其聚类内扩散标签
        for center_idx, center_label in center_labels.items():
            # 找到该中心所在的聚类
            target_cluster = None
            for i in range(1, len(clusters)):
                if center_idx in clusters[i]:
                    target_cluster = clusters[i]
                    break
            
            # 如果找到了聚类且该聚类没有被完全标注，则扩散标签
            if target_cluster is not None:
                # 检查该聚类是否已经被完全标注
                cluster_fully_labeled = True
                for idx in target_cluster:
                    is_labeled = False
                    for labeled_idx, _, _ in index_value_label_dict.get(attr, []):
                        if labeled_idx == idx:
                            is_labeled = True
                            break
                    if not is_labeled:
                        cluster_fully_labeled = False
                        break
                
                # 如果聚类没有被完全标注，则扩散标签
                if not cluster_fully_labeled:
                    for idx in target_cluster:
                        # 检查该索引是否已经被标注
                        is_labeled = False
                        for labeled_idx, _, _ in index_value_label_dict.get(attr, []):
                            if labeled_idx == idx:
                                is_labeled = True
                                break
                        
                        # 如果没有被标注，则添加到相应的列表
                        if not is_labeled:
                            if center_label == 1:
                                det_wrong_list.append((idx, attr))
                            elif center_label == 0:
                                det_right_list.append((idx, attr))
    
    return det_wrong_list, det_right_list


def process_gen_err_funcs(FUNC_USE, FUNC_READ, read_path, read_func_path, read_error_path, resp_path, funcs_directory, dirty_csv, all_attrs, para_file, related_attrs_dict, center_index_value_label_dict, det_wrong_list, det_right_list):
    err_gen_dict = defaultdict(default_dict_of_lists)
    funcs_for_attr = defaultdict(default_dict_of_lists)
    if FUNC_USE and FUNC_READ:
        for attr in all_attrs:
            file_names = os.listdir(read_func_path)
            for file_name in sorted(file_names):
                if file_name == f'funcs_zgen_{attr}.txt':
                    with open(os.path.join(read_func_path, file_name), 'r', encoding='utf-8') as file:
                        func_gen_str = file.read()
                    with open(os.path.join(funcs_directory, file_name), 'w', encoding='utf-8') as file:
                        file.write(func_gen_str)
                    try:
                        clean_flist, _ = extract_func('```python+\n' + func_gen_str + '\n```')
                        clean_flist = list(set(clean_flist))
                        funcs_for_attr[attr]['clean'].extend(clean_flist)
                    except Exception as e:
                        print(f"Error: {e}")
            para_file.write(f"{attr} func_num:{len(funcs_for_attr[attr]['clean'])}\n")

    elif FUNC_USE:
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(task_func_gen, attr, err_gen_dict) for attr in all_attrs]
            outputs = [result.result() for result in results]
            for output in outputs:
                funcs_for_attr.update(output)
    return err_gen_dict, funcs_for_attr


def process_gen_clean_funcs(PRE_FUNC_USE, PRE_FUNC_READ, read_pre_func_path, funcs_pre_directory, dirty_csv, all_attrs, related_attrs_dict, logger):
    pre_funcs_for_attr = defaultdict(default_dict_of_lists)
    if PRE_FUNC_USE and PRE_FUNC_READ:
        copy_read_files_in_dir(funcs_pre_directory, read_pre_func_path)
        for attr in all_attrs:
            file_names = os.listdir(read_pre_func_path)
            for file_name in sorted(file_names):
                if file_name == f'pre_funcs_zgen_{attr}.txt':
                    with open(os.path.join(read_pre_func_path, file_name), 'r', encoding='utf-8') as file:
                        func_gen_str = file.read()
                    try:
                        flist, _ = extract_func('```python+\n' + func_gen_str + '\n```')
                        flist = list(set(flist))
                        pre_funcs_for_attr[attr]['clean'].extend(flist)
                    except Exception as e:
                        print(f"Error: {e}")
    elif PRE_FUNC_USE:
        with ThreadPoolExecutor(max_workers=2*os.cpu_count()) as executor:
            results = [executor.submit(gen_clean_funcs, attr, dirty_csv, funcs_pre_directory, related_attrs_dict, logger) for attr in all_attrs]
            outputs = [result.result() for result in results]
            for output in outputs:
                pre_funcs_for_attr.update(output)
    elif not PRE_FUNC_USE:
        for attr in all_attrs:
            pre_funcs_for_attr[attr] = {'clean': []}
    return pre_funcs_for_attr


def gen_clean_funcs(attr, dirty_csv, funcs_pre_directory, related_attrs_dict, logger):  
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
    prompt = pre_func_prompt(attr, sample_rows_str)
    pre_func_response = get_ans_from_llm(prompt, api_use=API_USE)
    flist, _ = extract_func(pre_func_response)
    with open(os.path.join(funcs_pre_directory, f"prompt_pre_funcs_zgen_{attr}.txt"), 'w', encoding='utf-8') as prom_file:
        prom_file.write(prompt)
    with open(os.path.join(funcs_pre_directory, f"pre_funcs_zgen_{attr}.txt"), 'w', encoding='utf-8') as func_file:
        func_file.write("\n".join(list(set(flist))))
    funcs_for_attr = {attr: {'clean': flist}}
    return funcs_for_attr


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='run_config.yaml', help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Model settings
    FUNC_VAL_THRESHOLD = config['model']['func_val_threshold']
    n_method = config['model']['n_method']
    API_USE = config['model']['api_use']
    RELATED_ATTRS = config['model']['related_attrs']
    DISTRI_ANALYSIS = config['model']['distri_analysis']
    GUIDE_USE = config['model']['guide_use']
    PRE_FUNC_USE = config['model']['pre_func_use']
    FUNC_USE = config['model']['func_use']
    ENHANCED_USE = config['model']['enhanced_use']
    ERR_GEN_USE = config['model']['err_gen_use']
    REL_TOP = config['model']['rel_top']
    LABEL_PROP = config['model']['label_prop']
    ITERATIONS = config['model']['iterations']
    
    
    # Read settings
    PRE_FUNC_READ = config['read']['pre_func']
    DISTRI_ANALYSIS_READ = config['read']['distri_analysis']
    RELATED_ATTRS_READ = config['read']['related_attrs']
    CLUSTER_READ = config['read']['cluster']
    GUIDE_READ = config['read']['guide']
    ERROR_CHECKING_READ = config['read']['error_checking']
    EXTRA_ALL_LABEL = config['read']['extra_all_label']
    FUNC_READ = config['read']['func']
    ENHANCED_READ = config['read']['enhanced']
    ERR_GEN_READ = config['read']['err_gen']
    
    # Dataset settings
    base_dir = config['data']['base_dir']
    err_rate_list = config['data']['err_rate_list']
    all_set_num = config['data']['all_set_num']
    dataset_list = config['data']['datasets'] * all_set_num
    result_dir = config['data']['result_dir']
    dataset_list = sorted(dataset_list)
    set_num_list = [i % all_set_num + 1 for i in range(len(dataset_list))]
    
    err_check_val_num_per_query = config['data']['err_check_val_num_per_query']
    
    read_path_dict = config['read_paths']
    READ_IN_BATCH = config['read']['read_in_batch']
    if READ_IN_BATCH:
        read_path_dict = get_read_paths(config['read']['start_time'], config['read']['end_time'], base_dir)
    
    date_time = datetime.now().strftime("%m-%d")
    # 记录运行配置信息
    # run_info = config['model']['run_info']
    # info_path = f"{base_dir}/result/{result_dir}/{date_time} {run_info}"
    # os.makedirs(info_path, exist_ok=True)
    # with open(os.path.join(info_path, 'config.yaml'), 'w') as f:
    #     yaml.dump(config, f, default_flow_style=False)
    
    for set_num, dataset in zip(set_num_list, dataset_list):
        for err_rate in err_rate_list:
            read_path = ''
            if any([PRE_FUNC_READ, DISTRI_ANALYSIS_READ, RELATED_ATTRS_READ, CLUSTER_READ, GUIDE_READ, ERROR_CHECKING_READ, FUNC_READ]):
                read_path = read_path_dict[f'{dataset}{err_rate}-{set_num}']
            read_guide_path = os.path.join(read_path, 'guide_refine')
            read_error_checking_path = os.path.join(read_path, 'error_checking')
            read_func_path = os.path.join(read_path, 'funcs')
            read_enhanced_path = os.path.join(read_path, 'enhanced')
            read_pre_func_path = os.path.join(read_path, 'funcs_pre')
            read_err_gen_path = os.path.join(read_path, 'err_gen')
            read_error_path = read_path + 'funcs'
            
            date_time = datetime.now().strftime("%m-%d")
            now_time = datetime.now().strftime("%H:%M")
            resp_path = f"{base_dir}/result/{result_dir}/{date_time} {now_time} {dataset}{err_rate}-{n_method}-set{set_num}-iterations{ITERATIONS}"
            guide_directory = f'{resp_path}/guide'
            error_checking_res_directory = f'{resp_path}/error_checking'
            funcs_directory = f'{resp_path}/funcs'
            funcs_pre_directory = f'{resp_path}/funcs_pre'
            err_gen_directory = f'{resp_path}/err_gen'
            enhanced_gen_directory = f'{resp_path}/enhanced'
            os.makedirs(resp_path, exist_ok=True)
            os.makedirs(guide_directory, exist_ok=True)
            os.makedirs(error_checking_res_directory, exist_ok=True)
            os.makedirs(funcs_directory, exist_ok=True)
            os.makedirs(funcs_pre_directory, exist_ok=True)
            os.makedirs(err_gen_directory, exist_ok=True)
            os.makedirs(enhanced_gen_directory, exist_ok=True)
            
            dirty_path = base_dir + '/data/' + dataset + '_error-' + str(err_rate) + '.csv'
            clean_path = base_dir + '/data/' + dataset + '_clean.csv'
            clean_csv = pd.read_csv(clean_path, dtype=str).fillna('nan')
            dirty_csv = pd.read_csv(dirty_path, dtype=str).fillna('nan')
            all_attrs = list(dirty_csv.columns)
            
            para_file = open(os.path.join(resp_path, '0-parameters.txt'), 'w', encoding='utf-8')
            time_file = open(os.path.join(resp_path, '0-time.txt'), 'w', encoding='utf-8')
                    
            logger = Logger(resp_path).get_logger()
            
            parameters = {
                "executing File": os.path.abspath(__file__),
                "read_path": read_path,
                "resp_path": resp_path,
                "dirty_path": dirty_path,
                "clean_path": clean_path,
            }

            para_file.write("\n".join(f"{key}: {value}" for key, value in parameters.items()))
            para_file.write("\nConfig:\n")
            for section in ['model', 'read', 'data']:
                para_file.write(f"\n{section.title()}:\n")
                for key, value in config[section].items():
                    para_file.write(f"  {key}: {value}\n")
            para_file.write("\n")

            total_time = 0
            related_attrs_dict, gt_wrong_dict = {}, {}
            with Timer('Getting Related Attributes & gt_wrong_list', logger, time_file) as t:
                related_attrs_dict, gt_wrong_dict = process_related_attr(RELATED_ATTRS, RELATED_ATTRS_READ, REL_TOP, read_path, resp_path, clean_csv, dirty_csv, all_attrs)
            total_time += t.duration
                        
            pre_funcs_for_attr = {}
            with Timer('Preliminary Function Generation', logger, time_file) as t:
                pre_funcs_for_attr = process_gen_clean_funcs(PRE_FUNC_USE, PRE_FUNC_READ, read_pre_func_path, funcs_pre_directory, dirty_csv, all_attrs, related_attrs_dict, logger)
            total_time += t.duration
            
            cluster_index_dict, center_value_dict = {}, {}
            feature_all_dict = defaultdict(default_dict_of_lists)
            with Timer('Clustering', logger, time_file) as t:
                cluster_index_dict, center_value_dict, feature_all_dict = process_cluster(n_method, CLUSTER_READ, dataset, read_path, resp_path, dirty_csv, all_attrs, related_attrs_dict, pre_funcs_for_attr)
            total_time += t.duration
            
            distri_analy_content = {}
            with Timer('Analyzing Data Distribution', logger, time_file) as t:
                distri_analy_content = process_distri_analys(DISTRI_ANALYSIS, DISTRI_ANALYSIS_READ, read_path, resp_path, dirty_csv, all_attrs)
            total_time += t.duration
            
            guide_content = {}
            with Timer('Constructing Guidelines', logger, time_file) as t:
                guide_content = process_guidlines(GUIDE_USE, GUIDE_READ, dataset, read_path, read_guide_path, resp_path, dirty_csv, all_attrs, guide_directory, cluster_index_dict, distri_analy_content)
            total_time += t.duration
            iterations = ITERATIONS
            # 初始化labeled_number为0
            labeled_number = 0
            # 初始化index_value_label_dict
            index_value_label_dict = defaultdict(list)
            expert_labeled_number = 0
            num_epochs = 5000
            # 初始化之前选择过的聚类字典
            previously_selected_clusters = {}

            for i in range(iterations):
                indices_dict = {}                
                # 准备indices字典
                if i == 0:
                    # 第一次迭代使用聚类中心indices
                    indices_dict = {attr: clusters[0] for attr, clusters in cluster_index_dict.items()}
                else:
                    # 后续迭代使用最优聚类indices
                    indices_dict = {attr: optimal_cluster_result[attr]['cluster_indices'] for attr in all_attrs if attr in optimal_cluster_result}
                
                # 计算当前迭代的实际标注数量（过滤掉已标注的索引后）
                unlabeled_indices_dict = {}
                current_labeled_number = 0                
                for attr_name, indices in indices_dict.items():
                    # 获取该属性的未标注索引
                    if index_value_label_dict and attr_name in index_value_label_dict:
                        labeled_indices = {idx for idx, _, _ in index_value_label_dict[attr_name]}
                        unlabeled_indices = [idx for idx in indices if idx not in labeled_indices]
                    else:
                        unlabeled_indices = indices[:]                    
                    # 限制标注数量：超过30个随机抽取30个，否则全部标注
                    if i!=0:
                        if len(unlabeled_indices) > 30:
                            unlabeled_indices = random.sample(list(unlabeled_indices), 30)                    
                    unlabeled_indices_dict[attr_name] = unlabeled_indices
                    current_labeled_number += len(unlabeled_indices)                
                labeled_number += current_labeled_number
                
                with Timer('LLM Labeling', logger, time_file) as t:
                    if ERROR_CHECKING_READ:
                        process_error_checking(ERROR_CHECKING_READ, read_error_checking_path, all_attrs, error_checking_res_directory)
                    else:
                        # 使用未标注的索引进行标注
                        for attr_name, unlabeled_indices in unlabeled_indices_dict.items():
                            if len(unlabeled_indices) > 0:  # 只有当有未标注的索引时才进行标注
                                task_det_initial(attr_name, error_checking_res_directory, unlabeled_indices)
                total_time += t.duration
                
                with Timer('Extract Labeling Results', logger, time_file) as t:
                    if (EXTRA_ALL_LABEL):
                        current_index_value_label_dict = extract_llm_label_res(all_attrs, error_checking_res_directory)
                    else:
                        # 使用相同的indices字典提取标注结果
                        current_index_value_label_dict = extract_llm_label_res(all_attrs, error_checking_res_directory, indices_dict)
                    if i == 0:
                        center_index_value_label_dict = current_index_value_label_dict
                total_time += t.duration
                
                if i != 0:
                    # 用标注数据和增强数据训练模型并检测前面LLM标注不一致的数据用于后续数据的增强
                    feat_dict_train = {}
                    label_dict_train = {}
                    for attr in all_attrs:
                        # 在第一次迭代时，det_wrong_list和det_right_list为空，所以使用空列表
                        attr, feature_list, label_list = process_attr_train_feat(attr, dirty_csv, det_right_list, det_wrong_list, related_attrs_dict, enhanced_gen_dict, funcs_for_attr, feature_all_dict, resp_path)
                        feat_dict_train[attr] = feature_list
                        label_dict_train[attr] = label_list
                    
                    model_col = {}
                    
                    for attr in tqdm(all_attrs, desc="Training models", ncols=120):
                        attr, model, learning_rate, optimizer, model_str, epoch = train_model(attr, feat_dict_train[attr], label_dict_train[attr], num_epochs)
                        if model is not None:
                            model_col[attr] = model
                            
                    logger.info('Finish Generating Features & Training Models')
                    det_wrong_list_res = []
                    
                    for col, attr in tqdm(enumerate(all_attrs), desc="Making predictions", ncols=120):
                        wrong_cells = make_predictions(col, attr, dirty_csv, model_col, related_attrs_dict, funcs_for_attr, feature_all_dict, resp_path)
                        for cell in wrong_cells:
                            if cell not in det_wrong_list_res:
                                det_wrong_list_res.append(cell)
                    
                    # 比较LLM标注结果和分类器标注结果，找出标注不一致的索引
                    inconsistent_index_value_label_dict = compare_llm_and_classifier_labels(current_index_value_label_dict, det_wrong_list_res)
                    

                # 将当前迭代的标注结果累积到总结果中
                for attr, label_list in current_index_value_label_dict.items():
                    # 创建一个集合来跟踪已经添加的索引，避免重复
                    existing_indices = {idx for idx, _, _ in index_value_label_dict[attr]}
                    for idx, value, label in label_list:
                        # 只有当索引不存在时才添加
                        if idx not in existing_indices:
                            index_value_label_dict[attr].append((idx, value, label))
                            existing_indices.add(idx)
                
                measure_status = 'Not Done'
                with Timer('Evaluating LLM Labeling', logger, time_file) as t:
                    measure_status = measure_llm_label(resp_path, clean_csv, all_attrs, related_attrs_dict, gt_wrong_dict, index_value_label_dict)
                total_time += t.duration

                
                
                # 初始化det_wrong_list和det_right_list
                det_wrong_list = []
                det_right_list = []
                for attr, label_list in index_value_label_dict.items():
                    for idx, value, label in label_list:
                        if label == 1:
                            det_wrong_list.append((idx, attr))
                        elif label == 0:
                            det_right_list.append((idx, attr))
                
                if (i == 0):
                    err_gen_dict, funcs_for_attr = {}, {}
                    with Timer('Generating Functions', logger, time_file) as t:
                        err_gen_dict, funcs_for_attr = process_gen_err_funcs(FUNC_USE, FUNC_READ, read_path, read_func_path, read_error_path, resp_path, funcs_directory, dirty_csv, all_attrs, para_file, related_attrs_dict, center_index_value_label_dict, det_wrong_list, det_right_list)
                    total_time += t.duration
                

                with Timer('Generating Enhanced Data', logger, time_file) as t:
                    # 如果是第一次迭代，初始化enhanced_gen_dict
                    if i == 0:
                        enhanced_gen_dict = defaultdict(default_dict_of_lists)
                        # 用这些标注的聚类中心增强数据，使用wrong_values和right_values
                        for attr in all_attrs:
                            # 从current_index_value_label_dict获取wrong_values和right_values
                            wrong_values, right_values = prepare_wrong_right_values(attr, current_index_value_label_dict)
                            
                            # 调用生成增强数据的方法
                            generate_enhanced_data_from_wrong_right_values(
                                attr, dirty_csv, clean_csv, related_attrs_dict, enhanced_gen_dict, 
                                wrong_values, right_values, 15
                            )
                    else:
                        # 在主循环的else分支下获取wrong_values、right_values和actual_right_values
                        for attr in all_attrs:
                            wrong_values, right_values, actual_right_values = prepare_enhanced_data_values(
                                attr, dirty_csv, clean_csv, inconsistent_index_value_label_dict, related_attrs_dict, index_value_label_dict
                            )
                            
                            # 调用生成增强数据的方法
                            generate_enhanced_data_from_values(
                                attr, dirty_csv, clean_csv, related_attrs_dict, enhanced_gen_dict, 
                                wrong_values, right_values, actual_right_values, 15
                            )


                # 计算并记录inconsistent_index_value_label_dict中被标注为错误的数据数量
                if (i != 0):
                    expert_labeled_number = 0
                    for attr in all_attrs:
                        if attr in inconsistent_index_value_label_dict:
                            expert_labeled_number += len(inconsistent_index_value_label_dict[attr])
                    # 记录到参数文件
                    para_file.write(f"第 {i+1} 次迭代中专家标注的数量: {expert_labeled_number}\n")

                total_time += t.duration
                if (i != iterations-1):
                    with Timer('Select Optimal Cluster', logger, time_file) as t:
                        optimal_cluster_result = process_select_optimal_cluster(
                            enhanced_gen_dict, cluster_index_dict, dirty_csv, all_attrs, related_attrs_dict, 
                            pre_funcs_for_attr, resp_path, logger, index_value_label_dict, 
                            residual_method='both', previously_selected_clusters=previously_selected_clusters
                        )
                        # 更新之前选择过的聚类字典
                        for attr, cluster_info in optimal_cluster_result.items():
                            if attr not in previously_selected_clusters:
                                previously_selected_clusters[attr] = []
                            previously_selected_clusters[attr].append(cluster_info['cluster_idx'])
                    total_time += t.duration


            para_file.write(f"LLM labeled value number: {labeled_number}\n")

            # 保存每列增强的干净数据和脏数据的数量
            para_file.write("\nEnhanced Data Statistics:\n")
            for attr, data in enhanced_gen_dict.items():
                clean_count = len(data['clean'])
                dirty_count = len(data['dirty'])
                para_file.write(f"{attr}: clean_data_count={clean_count}, dirty_data_count={dirty_count}\n")
            with Timer('Label Propagation', logger, time_file) as t:
                det_wrong_list, det_right_list = label_prop(resp_path, dirty_path, clean_path, cluster_index_dict, index_value_label_dict, LABEL_PROP)
            total_time += t.duration
            
            

            feature_all_dict = None
            if os.path.exists(os.path.join(resp_path, f'cluster_feat_dict.pkl')):
                with open(os.path.join(resp_path, f'cluster_feat_dict.pkl'), 'rb') as f:
                    feature_all_dict = pickle.load(f)
                    
            
            logger.info('Start Training Local Models')
            time_start = time.time()
            # 改为单线程处理
            feat_dict_train = {}
            label_dict_train = {}
            for attr in all_attrs:
                attr, feature_list, label_list = process_attr_train_feat(attr, dirty_csv, det_right_list, det_wrong_list, related_attrs_dict, enhanced_gen_dict, funcs_for_attr, feature_all_dict, resp_path)
                feat_dict_train[attr] = feature_list
                label_dict_train[attr] = label_list
            
            model_col = {}
            
            for attr in tqdm(all_attrs, desc="Training models", ncols=120):
                attr, model, learning_rate, optimizer, model_str, epoch = train_model(attr, feat_dict_train[attr], label_dict_train[attr], num_epochs)
                if model is not None:
                    model_col[attr] = model
                    
            logger.info('Finish Generating Features & Training Models')
            para_file.close()
            det_wrong_list_res = []
            
            for col, attr in tqdm(enumerate(all_attrs), desc="Making predictions", ncols=120):
                wrong_cells = make_predictions(col, attr, dirty_csv, model_col, related_attrs_dict, funcs_for_attr, feature_all_dict, resp_path)
                for cell in wrong_cells:
                    if cell not in det_wrong_list_res:
                        det_wrong_list_res.append(cell)
                        
            det_res_path = os.path.join(resp_path, "func_det_res.txt")
            measure_detect(clean_path, dirty_path, list(det_wrong_list_res), det_res_path)
            time_end = time.time()
            logger.info(f'Finish Local Model Training and Prediction, Using {time_end - time_start}s')
            total_time += time_end - time_start
            time_file.write(f"model_training: {time_end - time_start}\n")
            time_file.write(f"total: {total_time}\n")
        time_file.close()
        para_file.close()