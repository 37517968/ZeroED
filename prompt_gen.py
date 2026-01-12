
import re
import json
import random

def kb_gen_prompt(attr_name, dataset_name, idx_list, dirty_csv, attr_analy_content):
    prompt = f'You are a top data scientist, especially in data cleaning. Please generate a comprehensive and expert guide for identifying and analyzing common errors in the \'{attr_name}\' attribute of the \'{dataset_name}\' table:'
    if len(attr_analy_content) > 0:
        prompt += f"\n\nHere are the data distribution analysis results for the attribute \'{attr_name}\':"
        prompt += attr_analy_content
    prompt += f'\n\nHere are some examples for \'{attr_name}\' along with strong correlated attribute values:\n'
    
    max_example_num = 20
    idx_list = idx_list[:max_example_num] if len(idx_list) > max_example_num else idx_list
    random.shuffle(idx_list)
    example_vals = []
    for idx in idx_list:
        example_vals.append(str(dirty_csv.loc[int(idx), :].to_dict()))
    example_vals_str = '\n'.join(example_vals)
    prompt += example_vals_str
    
    prompt += f'\n\nPlease first explain the meaning of attribute \'{attr_name}\'.\n'
    prompt += f'\n\nThen, for each error type below, considering the data distribution analysis results, provide specific causes, examples, and detection methods for \'{attr_name}\':\n'
    prompt += '1. Pattern Violations: Expected formats and non-conforming value identification.\n'
    prompt += '2. Missing Values: Explicit null indicators and implicit missing data patterns.\n'
    prompt += f'3. Constraint Violations: Relationships of \'{attr_name}\' with other attributes and how to verify them.\n'
    prompt += '4. Out-of-domain Values: Valid value range/set and possible outliers.\n'
    prompt += '5. Typos: Common misspellings or data entry errors and detection strategies.\n'
    prompt += '6. Common Knowledge Violations: Expected rules/facts and methods to identify contradictions.\n\n'
    prompt += f'Please also generate some possible errors for the \'{attr_name}\' attribute data based on the above error types. '
    prompt += '\n\nIMPORTANT NOTE: When analyzing potential errors, if you are not completely certain that a value is wrong, please respect the mainstream data distribution patterns. Some values that appear unusual may actually be valid according to local requirements or domain-specific conventions. Only flag values as errors when you have high confidence they violate clear patterns or rules.'
    return prompt


def error_check_prompt(col_values, col_name, expert_labeled_right_dict, expert_labeled_wrong_dict):
    lines = col_values.strip().split('\n')
    try:
        col_list = re.findall(r'"([^"]+)"\s*:', lines[0])
    except json.JSONDecodeError as e:
       print(f"JSON Decode Error: {e}")
       print(f"Problematic JSON string: {lines[0]}")

    template_dict_1 = {key: f'{key}_example_val_1' for key in col_list}
    template_dict_2 = {key: f'{key}_example_val_2' for key in col_list}
    
    prompt = ""
    prompt += f"As a data quality expert, please first analyze attribute relations and analyze the '{col_name}' attribute values for potential errors. \n"
    prompt += "-----------------------------------------------\n\n"
    prompt += "Here are the given inputs:\n"
    prompt += f"Values of column '{col_name}' along with related attribute values:\n"
    prompt += f"'{col_values}'\n"
    prompt += f"Provide your analysis on `{col_name}` values in JSON format as follows, **do not care problems in other attributes**:\n\n"
    prompt += '''
```json
{'''
    prompt += f'''"column_name": "{col_name}",'''
    prompt += '''
  "entries": [
    {'''
    prompt += f'''\n"value_row": "{template_dict_1}",'''
    prompt += f'''\n"error_analysis": "[Brief explanation of the error analysis, if applicable]",'''
    prompt += f'''\n"has_error_in_{col_name}_value": true/false,'''
    prompt += '''
    },
    {'''
    prompt += f'''\n"value_row": "{template_dict_2}",'''
    prompt += f'''\n"error_analysis": "[Brief explanation of the error analysis, if applicable]",'''
    prompt += f'''\n"has_error_in_{col_name}_value": true/false,'''
    prompt += '''
    }
  ]
}
```
\n\n'''
    prompt += "- You MUST strictly follow all the rules below.\n\n"
    prompt += "- Only mark a value as an error if you are confident it is incorrect.\n"
    prompt += "- Do NOT mark a value as an error solely because it is not present in examples.\n"
    prompt += "- Ignore the case sensitivity issues.\n"
    prompt += "- Do not check for data type errors.\n\n"
    if col_name in expert_labeled_right_dict or col_name in expert_labeled_wrong_dict:
        prompt += f"Below are reference examples for analyzing the correctness of `{col_name}` values.\n"
        prompt += "**These examples illustrate patterns only. They are NOT an exhaustive list.**\n"
        prompt += "**Do NOT mark a value as wrong simply because it does not appear in the examples.**\n\n"

    if col_name in expert_labeled_right_dict:
        prompt += "### Valid example patterns:\n"
        prompt += json.dumps(expert_labeled_right_dict[col_name], indent=2, ensure_ascii=False)
        prompt += "\n\n"

    if col_name in expert_labeled_wrong_dict:
        prompt += "### Wrong example patterns:\n"
        prompt += json.dumps(expert_labeled_wrong_dict[col_name], indent=2, ensure_ascii=False)
        prompt += "\n\n"


    return prompt


def create_dirty_gen_inst_prompt(clean_vals, clean_vals_sample, dirty_vals_sample, target_attribute, num_errors=20):
    if len(clean_vals) > 0:
        temp_vals = clean_vals[0]
    elif len(clean_vals_sample) > 0:
        temp_vals = clean_vals_sample[0]
    elif len(dirty_vals_sample) > 0:
        temp_vals = dirty_vals_sample[0]
    else:
        print(f"No vals in clean_vals and dirty_vals_sample of attr {target_attribute}")
        temp_vals = f"{target_attribute}: none"
    attrs = re.findall(r"'(\w+)':", str(temp_vals))
    template_dict_1 = {key: f'{key}_val_1' for key in attrs}
    template_dict_1[target_attribute] = 'dirty_value_1'
    template_dict_2 = {key: f'{key}_val_2' for key in attrs}
    template_dict_2[target_attribute] = 'dirty_value_2'
    
    prompt = f"""
You are a data quality analyst. Your task is to inject realistic errors into clean data for the attribute `{target_attribute}`.
----------------------------------------------------------
### 1. Learn Error Patterns
You are provided with **paired examples** of:
- Clean samples:{clean_vals_sample}
- Dirty samples:{dirty_vals_sample}

From each pair, infer the transformation rules (error type and how it changes the clean value).
----------------------------------------------------------
### 2. Generate New Errors
For the following clean values:
{clean_vals}

Generate **{num_errors} erroneous versions per clean value**, ensuring:
- For each clean value, generate a corresponding erroneous value (one-to-one mapping), rather than reusing or copying any provided dirty sample.
- Errors follow the learned transformation patterns.
- Each error value differs from the clean value.
- The general data type and structure remain valid.
----------------------------------------------------------
### 3. Output Format (strict)
The output should be in the following strict format:
['{target_attribute}', error_value_1, Reason: 'Error type1: Specific reason', {str(template_dict_1)}]
['{target_attribute}', error_value_2, Reason: 'Error type2: Specific reason', {str(template_dict_2)}]
...
Where:
- `error_value` is the newly generated erroneous value.
- `Reason` describes the applied transformation.
--------------------------------------------------------------------------
"""
    return prompt

def create_clean_gen_inst_prompt(clean_vals, target_attribute, num_gen=20):
    if len(clean_vals) > 0:
        temp_vals = clean_vals[0]
    else:
        print(f"No vals in clean_vals of attr {target_attribute}")
        temp_vals = f"{target_attribute}: none"
    attrs = re.findall(r"'(\w+)':", str(temp_vals))
    template_dict_1 = {key: f'{key}_val_1' for key in attrs}
    template_dict_1[target_attribute] = 'clean_value_1'
    template_dict_2 = {key: f'{key}_val_2' for key in attrs}
    template_dict_2[target_attribute] = 'clean_value_2'

    prompt = f"""
You are a data quality analyst with extensive experience in generating realistic clean data. Your task is to analyze a given dataset and generate plausible clean values for a specific attribute, following the same distribution and patterns as the provided examples.

I will provide you with a sample of **clean** values in a tabular format for various attributes. Your objectives are to:

1. Analyze the data to identify patterns, relationships, and constraints between attributes.
2. Focus on the attribute named `{target_attribute}` and generate realistic clean values that follow the same distribution.
3. Ensure the clean values you generate are diverse but consistent with the data patterns.

Your task is to analyze the data and identify inner relationships. Based on this analysis, generate clean values specifically for the attribute `{target_attribute}` that follow the same patterns and distribution as the examples.
"""
    if clean_vals:
        prompt += f"For the attribute `{target_attribute}`, here are the given **clean** tuples as examples:\n"
        prompt += '\n'.join([str(i) for i in clean_vals]) + '\n\n'
    prompt += f"Please analyze the data patterns and generate {num_gen} realistic clean values specifically for the attribute `{target_attribute}`:\n"
    prompt += f"""
The output should be in the following strict format:
['{target_attribute}', clean_value_1, Pattern description: 'Specific description', {str(template_dict_1)}]
['{target_attribute}', clean_value_2, Pattern description: 'Specific description', {str(template_dict_2)}]
...
Please ensure that the descriptions for each clean value are clearly specified, explaining the pattern it follows.
Do not duplicate the reference values exactly, but create new values that follow the same distribution.
--------------------------------------------------------------------------
"""
    return prompt

def create_err_gen_inst_prompt(clean_vals, dirty_vals, target_attribute, num_errors=20):
    if len(clean_vals) > 0:
        temp_vals = clean_vals[0]
    elif len(dirty_vals) > 0:
        temp_vals = dirty_vals[0]
    else:
        print(f"No vals in clean_vals and dirty_vals of attr {target_attribute}")
        temp_vals = f"{target_attribute}: none"
    attrs = re.findall(r"'(\w+)':", str(temp_vals))
    template_dict_1 = {key: f'{key}_val_1' for key in attrs}
    template_dict_1[target_attribute] = 'error_value_1'
    template_dict_2 = {key: f'{key}_val_2' for key in attrs}
    template_dict_2[target_attribute] = 'error_value_2'
    
    prompt = f"""
You are a data quality analyst with extensive experience in identifying and generating realistic data errors. Your task is to analyze a given dataset and generate plausible errors for a specific attribute, simulating real-world data quality issues.

I will provide you with a sample of **possible** clean and dirty values in a tabular format for various attributes. Your objectives are to:

1. Analyze the data to identify patterns, relationships, and constraints between attributes.
2. Focus on the attribute named `{target_attribute}` and generate realistic errors that could occur in real-world scenarios.
3. Ensure the errors you generate are diverse and cover multiple error types.

Your task is to analyze the data and identify inner relationships. Based on this analysis, generate errors specifically for the attribute `attribute_name` as they might occur in real-world scenarios. 
The types of errors include the following ones
1. Pattern Violations: Values that don't match the expected format
2. Explicit/Implicit Missing Values: Null values or placeholders for missing data
3. Constraints Violations: Values that conflict with other columns or violate business rules
4. Out-of-domain values: Values outside the expected range or set
5. Typos: Spelling or data entry errors
6. Violate common knowledge: Values that contradict widely known facts
"""
    prompt += f"For the attribute `{target_attribute}`, here are the given **possible** clean tuples:\n"
    prompt += '\n'.join([str(i) for i in clean_vals]) + '\n'
    prompt += f"There are also some **possible** wrong tuples for reference:\n"
    prompt += '\n'.join([str(i) for i in dirty_vals]) + '\n\n'
    prompt += f"Please analyze the error pattern and generate {num_errors} realistic errors specifically for the attribute `{target_attribute}`:\n"
    prompt += f"""
The output should be in the following strict format:
['{target_attribute}', error_value_1, Reason: 'Error type1: Specific reason', {str(template_dict_1)}]
['{target_attribute}', error_value_2, Reason: 'Error type2: Specific reason', {str(template_dict_2)}]
...
Please ensure that the reasons for each error are clearly specified.
Do not be the same as the reference values.
--------------------------------------------------------------------------
"""
    return prompt


def pre_func_prompt(attr_name, data_example):
    prompt = (
        f"You are a Data Cleaning Specialist tasked with distinguishing between clean and dirty cells in the `{attr_name}`.\n\n"
        
        f"Here are examples for the '{attr_name}' column:\n"
        f"{data_example}\n\n"

        "Your task:\n"
        f"1. Analyze the `{attr_name}` column values.\n"
        "2. Create precise judge functions in Python that:\n"
        f"- Receive the row content of the `{attr_name}` column\n"
        "- Return True for clean values, False for dirty values\n"
        "- Use the naming convention 'is_clean_[judgment]'\n"
        "- Cover different perspectives of cleanliness as possible\n"
        "- Do not contain blank lines inner functions\n\n"

"Example function code snippet:\n"
"```python "
f"def is_clean_[judgment](row, attr):\n"
f"    # Value of `{attr_name}` is row[attr]\n"
"    # Your logic here\n"
"    return True  # or False\n"
"```\n"
"Provide your functions below:\n"
    )
    return prompt


def err_clean_func_prompt(attr_name, clean_info, errs_info):
    prompt = (
        f"You are a Data Cleaning Specialist tasked with identifying and distinguishing between clean and dirty cells in the `{attr_name}` column.\n\n"
        f"Clean examples for the '{attr_name}' column:\n"
        f"{clean_info}\n\n"
        f"Error examples for the '{attr_name}' column:\n"
        f"{errs_info}\n\n"

        "Your task:\n"
        f"1. Analyze the `{attr_name}` column values.\n"
        "2. Compare the differences between clean and dirty values.\n"
        "3. Create precise judge functions in Python that:\n"
        f"- Receive the row content of the `{attr_name}` column\n"
        "- Return True for clean values, False for dirty values\n"
        "- Use the naming convention 'is_clean_[judgment]'\n"
        "- Cover different perspectives of cleanliness as possible\n"
        "- Do not contain blank lines inner functions\n\n"

"Example function code snippet:\n"
"```python "
f"def is_clean_[judgment](row, attr):\n"
f"    # Value of `{attr_name}` is row[attr]\n"
"    # Your logic here\n"
"    return True  # or False\n"
"```\n"
"Provide your functions below:\n"
    )
    return prompt

def guide_gen_prompt():
    return

def distribution_analysis_decision_prompt(attr_name, cluster_center_values):
    """
    生成让LLM判断是否需要调用分布分析方法的prompt
    
    Args:
        attr_name: 属性名称
        cluster_center_values: 聚类中心的代表性值列表
    """
    values_str = '\n'.join([f"- {v}" for v in cluster_center_values])
    
    prompt = f"""You are a data quality expert. Below are representative values from the '{attr_name}' column that were obtained through clustering analysis.

Representative values for column '{attr_name}':
{values_str}

These values represent different patterns found in the data. Your task is to determine whether you can confidently distinguish between correct and incorrect values.

If you observe multiple distinct patterns in these values and you are uncertain which pattern(s) represent the correct/canonical form, you should request distribution analysis to help identify the dominant patterns.

Question: Can you accurately distinguish correct values from incorrect values based on these representative samples? If there are multiple patterns that make you uncertain about which pattern is correct, you may choose to call the distribution analysis method.

Please respond with ONLY 'yes' or 'no':
- 'yes' means you need distribution analysis (multiple patterns exist and you're uncertain)
- 'no' means you can confidently identify correct/incorrect values without distribution analysis
"""
    return prompt


def canonical_pattern_analysis_prompt(attr_name, cluster_samples, cluster_id, canonical_score):
    """
    生成让LLM分析标准模式的prompt
    
    Args:
        attr_name: 属性名称
        cluster_samples: 该簇中的样本值列表（最多10条）
        cluster_id: 簇的ID
        canonical_score: 该簇的规范得分
    """
    samples_str = '\n'.join([f"- {s}" for s in cluster_samples])
    
    prompt = f"""You are a data quality expert specializing in pattern recognition. Analyze the following sample values from a data cluster to identify the canonical (standard) pattern.

Column: '{attr_name}'
Cluster ID: {cluster_id}
Canonical Score: {canonical_score:.4f}

Sample values from this cluster:
{samples_str}

Your task:
1. Identify the common pattern shared by these values
2. Describe the canonical/standard format for this pattern
3. Provide a regex pattern (if applicable) that matches valid values
4. List key characteristics that define this pattern

Please respond in the following JSON format:
```json
{{
    "pattern_name": "A short descriptive name for this pattern",
    "pattern_description": "Detailed description of the canonical pattern",
    "regex_pattern": "Regular expression pattern (or 'N/A' if not applicable)",
    "key_characteristics": ["characteristic1", "characteristic2", ...],
    "example_valid_values": ["example1", "example2", ...],
    "common_errors": ["potential error type 1", "potential error type 2", ...]
}}
```
"""
    return prompt


def error_check_with_canonical_prompt(col_values, col_name, expert_labeled_right_dict, expert_labeled_wrong_dict, canonical_patterns):
    """
    带有标准模式上下文的错误检查prompt
    
    Args:
        col_values: 待检查的列值
        col_name: 列名
        expert_labeled_right_dict: 专家标注的正确样本字典
        expert_labeled_wrong_dict: 专家标注的错误样本字典
        canonical_patterns: 标准模式列表
    """
    lines = col_values.strip().split('\n')
    try:
        col_list = re.findall(r'"([^"]+)"\s*:', lines[0])
    except json.JSONDecodeError as e:
       print(f"JSON Decode Error: {e}")
       print(f"Problematic JSON string: {lines[0]}")

    template_dict_1 = {key: f'{key}_example_val_1' for key in col_list}
    template_dict_2 = {key: f'{key}_example_val_2' for key in col_list}
    
    prompt = ""
    prompt += f"As a data quality expert, please analyze the '{col_name}' attribute values for potential errors. \n"
    prompt += "-----------------------------------------------\n\n"
    
    # 添加标准模式上下文
    if canonical_patterns and len(canonical_patterns) > 0:
        prompt += f"### Canonical Patterns for '{col_name}':\n"
        prompt += "The following are the identified standard/canonical patterns for this column. Values that deviate significantly from these patterns may be errors.\n\n"
        for i, pattern in enumerate(canonical_patterns, 1):
            prompt += f"**Pattern {i}: {pattern.get('pattern_name', 'Unknown')}**\n"
            prompt += f"- Description: {pattern.get('pattern_description', 'N/A')}\n"
            if pattern.get('regex_pattern') and pattern.get('regex_pattern') != 'N/A':
                prompt += f"- Regex: `{pattern.get('regex_pattern')}`\n"
            if pattern.get('key_characteristics'):
                prompt += f"- Key characteristics: {', '.join(pattern.get('key_characteristics', []))}\n"
            prompt += "\n"
        prompt += "-----------------------------------------------\n\n"
    
    prompt += "Here are the given inputs:\n"
    prompt += f"Values of column '{col_name}' along with related attribute values:\n"
    prompt += f"'{col_values}'\n"
    prompt += f"Provide your analysis on `{col_name}` values in JSON format as follows, **do not care problems in other attributes**:\n\n"
    prompt += '''
```json
{'''
    prompt += f'''"column_name": "{col_name}",'''
    prompt += '''
  "entries": [
    {'''
    prompt += f'''\n"value_row": "{template_dict_1}",'''
    prompt += f'''\n"error_analysis": "[Brief explanation of the error analysis, if applicable]",'''
    prompt += f'''\n"has_error_in_{col_name}_value": true/false,'''
    prompt += '''
    },
    {'''
    prompt += f'''\n"value_row": "{template_dict_2}",'''
    prompt += f'''\n"error_analysis": "[Brief explanation of the error analysis, if applicable]",'''
    prompt += f'''\n"has_error_in_{col_name}_value": true/false,'''
    prompt += '''
    }
  ]
}
```
\n\n'''
    prompt += "- You MUST strictly follow all the rules below.\n\n"
    prompt += "- Only mark a value as an error if you are confident it is incorrect.\n"
    prompt += "- Use the canonical patterns above as reference for identifying errors.\n"
    prompt += "- Do NOT mark a value as an error solely because it is not present in examples.\n"
    prompt += "- Ignore the case sensitivity issues.\n"
    prompt += "- Do not check for data type errors.\n\n"
    
    if col_name in expert_labeled_right_dict or col_name in expert_labeled_wrong_dict:
        prompt += f"Below are reference examples for analyzing the correctness of `{col_name}` values.\n"
        prompt += "**These examples illustrate patterns only. They are NOT an exhaustive list.**\n"
        prompt += "**Do NOT mark a value as wrong simply because it does not appear in the examples.**\n\n"

    if col_name in expert_labeled_right_dict:
        prompt += "### Valid example patterns:\n"
        prompt += json.dumps(expert_labeled_right_dict[col_name], indent=2, ensure_ascii=False)
        prompt += "\n\n"

    if col_name in expert_labeled_wrong_dict:
        prompt += "### Wrong example patterns:\n"
        prompt += json.dumps(expert_labeled_wrong_dict[col_name], indent=2, ensure_ascii=False)
        prompt += "\n\n"

    return prompt


def llm_canonicality_score_prompt(attr_name, sample_values):
    """
    生成让LLM判断聚类样本值规范性的prompt
    
    Args:
        attr_name: 属性名称
        sample_values: 样本值列表（最多5个）
    """
    values_str = '\n'.join([f"- \"{v}\"" for v in sample_values])
    
    prompt = f"""You are a data quality expert. Please evaluate the canonicality (validity/correctness) of the following sample values from the '{attr_name}' column.

Sample values:
{values_str}

Please evaluate these values based on the following criteria:
1. Are these values meaningful and valid data? (not empty, null, placeholder, or obviously erroneous)
2. Do these values follow a reasonable format for this type of data?
3. Are these values likely to be correct/canonical representations?

Score guidelines:
- 0.0-0.2: Obviously invalid (empty strings, null, 'nan', 'N/A', placeholder values, garbage data)
- 0.2-0.4: Likely invalid (suspicious patterns, incomplete data, obvious errors)
- 0.4-0.6: Uncertain (could be valid or invalid, ambiguous)
- 0.6-0.8: Likely valid (reasonable format, plausible values)
- 0.8-1.0: Highly valid (clear, well-formatted, canonical values)

Please respond with ONLY a single decimal number between 0.0 and 1.0 representing the canonicality score.
Do not include any explanation, just the number.
"""
    return prompt
