
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
    prompt += "- Do not check for data type errors.\n"
    prompt += "- Null values are also errors.\n"
    prompt += "- **IMPORTANT for temporal data (dates, times, years, etc.)**: When evaluating date/time values, you MUST consider the context and use your knowledge to judge whether the value is reasonable. For example:\n"
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
3. Create a Python function that can check if a value matches this pattern
4. List key characteristics that define this pattern

Please respond in the following JSON format:
```json
{{
    "pattern_name": "A short descriptive name for this pattern",
    "pattern_description": "Detailed description of the canonical pattern",
    "pattern_function": "def matches_pattern(value):\\n    # Python function code here\\n    # Return True if value matches this pattern, False otherwise\\n    return True",
    "key_characteristics": ["characteristic1", "characteristic2", ...],
    "example_valid_values": ["example1", "example2", ...],
    "common_errors": ["potential error type 1", "potential error type 2", ...]
}}
```

IMPORTANT: The pattern_function should be a complete Python function that:
- Takes a single parameter 'value' (string)
- Returns True if the value matches the pattern, False otherwise
- Uses proper Python syntax with \\n for newlines
- Handles edge cases (empty strings, None, etc.)
- Is self-contained (no external dependencies except standard library like re, datetime, etc.)
- Function name must be 'matches_pattern'

Example pattern_function:
"def matches_pattern(value):\\n    value = value.strip()\\n    # Check if value matches the pattern\\n    return len(value) > 0 and value.isdigit()"
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
            # 不在提示词中显示pattern_function，只显示描述和特征
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
    prompt += "- Do not check for data type errors.\n"
    prompt += "- **IMPORTANT for temporal data (dates, times, years, etc.)**: When evaluating date/time values, you MUST consider the context and use your knowledge to judge whether the value is reasonable. "
    
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


def llm_compare_patterns_canonicality_prompt(attr_name, patterns_with_samples):
    """
    生成让LLM比较多个规范并给出规范性分数的prompt
    
    Args:
        attr_name: 属性名称
        patterns_with_samples: 列表，每个元素是 (pattern_description, sample_values, cluster_size)
    """
    patterns_str = ""
    for i, (pattern_desc, samples, cluster_size) in enumerate(patterns_with_samples, 1):
        # 格式化样本展示（最多3个不同的样本）
        if len(samples) == 0:
            samples_str = "(no samples)"
        elif len(samples) == 1:
            samples_str = f'"{samples[0]}"'
        elif len(samples) == 2:
            samples_str = f'"{samples[0]}", "{samples[1]}"'
        else:
            samples_str = ', '.join([f'"{s}"' for s in samples])
        
        patterns_str += f"\n**Pattern {i}** (Cluster size: {cluster_size}):\n"
        patterns_str += f"Description: {pattern_desc}\n"
        patterns_str += f"Sample values (up to 3 diverse examples): {samples_str}\n"
    
    prompt = f"""You are a data quality expert. Please evaluate and compare the following patterns found in the '{attr_name}' column.

{patterns_str}

Your task:
1. Evaluate the canonicality (validity/correctness) of each pattern
2. If there are conflicting patterns, identify which one(s) are more reasonable/canonical
3. Assign a canonicality score (0.0-1.0) to each pattern

Scoring guidelines:
- **Conflicting patterns**: If patterns conflict with each other, give higher scores (0.7-1.0) to the more reasonable/canonical pattern(s), and lower scores (0.0-0.3) to the less reasonable ones
    - Examples: "12.0 oz" vs "12.0 oz.", "12.0 oz" is more canonical
                "eng" vs "English", "eng" is more canonical
- **Invalid patterns**: Empty values, null, 'nan', 'N/A', placeholder values should get very low scores (0.0-0.2)
- **Valid patterns**: Clear, well-formatted, canonical patterns should get high scores (0.7-1.0)
- **Uncertain patterns**: Ambiguous or questionable patterns should get medium scores (0.3-0.6)

Please respond in the following JSON format:
```json
{{
  "pattern_1": {{
    "score": 0.0-1.0,
    "reasoning": "Brief explanation"
  }},
  "pattern_2": {{
    "score": 0.0-1.0,
    "reasoning": "Brief explanation"
  }},
  ...
}}
```

Only respond with the JSON, no additional text.
"""
    return prompt


def error_pattern_incompatibility_prompt(attr_name, canonical_samples, error_candidate_samples):
    """
    生成让LLM评估错误候选模式与canonical模式对立程度的prompt
    
    修改：不使用聚类描述，只使用样本值进行比较
    
    区分描述性列和格式类列：
    - 描述性列：语义不同但都合理的表述可以共存，不兼容分数较低
    - 格式类列：能表达相同语义但格式不一致，不兼容分数较高
    
    Args:
        attr_name: 属性名称
        canonical_samples: canonical模式的示例值列表（最多5个）
        error_candidate_samples: 错误候选模式的样本值列表（最多5个）
    """
    canonical_samples_str = '\n'.join([f'- "{s}"' for s in canonical_samples])
    candidate_samples_str = '\n'.join([f'- "{s}"' for s in error_candidate_samples])
    
    prompt = f"""
You are a data quality expert.

Your task is to assign an INCOMPATIBILITY or ERROR score (0.0–1.0) between a candidate pattern and the canonical pattern for column "{attr_name}".

Compare ONLY the sample values below. 

---

Canonical Pattern Examples:
{canonical_samples_str}

Candidate Pattern Examples:
{candidate_samples_str}

---

Scoring Principles (IMPORTANT):

1. High Incompatibility (0.8–1.0):
- Assign ONLY when there is a clear, structural/formatting mismatch in the pattern itself.
- Key indicators: Systematic differences in unit presentation (e.g., “oz” vs. “ounce”), presence/absence of key punctuation (e.g., “oz” vs. “oz.”).
- Identify punctuation errors where the text is enclosed in brackets [like this] while the correct pattern not.
- Don't check for spelling error, just only check for pattern error.

2. Medium Incompatibility (0.3–0.5):
- Assign when the values represent different but potentially valid expressions within a descriptive field.
- Applies to columns like *_title, *_abbreviation, *_name, article_pagination, journal_issn (if containing dates like “Jan-55”), beer_style.


3. Low Incompatibility (0.1–0.2):
- Assign when patterns are essentially consistent, with only minor, ignorable variations.
- Key indicators: Case differences (e.g., “ENG” vs “eng”), insignificant whitespace, or different lengths of sequential IDs.


---

Return ONLY a single decimal number between 0.0 and 1.0.
Do NOT provide explanations.
"""

    return prompt



def generate_cluster_descriptions_prompt(attr_name, clusters_with_samples):
    """
    生成让LLM为多个聚类生成自然语言描述的prompt
    
    Args:
        attr_name: 属性名称
        clusters_with_samples: 列表，每个元素是 (cluster_idx, sample_values, cluster_size)
    
    Returns:
        prompt: 提示词字符串
    """
    clusters_str = ""
    for i, (cluster_idx, samples, cluster_size) in enumerate(clusters_with_samples, 1):
        # 格式化样本展示（最多5个不同的样本）
        if len(samples) == 0:
            samples_str = "(no samples)"
        else:
            samples_str = '\n    '.join([f'- "{s}"' for s in samples])
        
        clusters_str += f"\n**Cluster {i}** (ID: {cluster_idx}, Size: {cluster_size}):\n"
        clusters_str += f"  Sample values:\n    {samples_str}\n"
    
    prompt = f"""You are a data quality expert specializing in pattern recognition. Analyze the following clusters of values from the '{attr_name}' column.

{clusters_str}

Your task:
1. For EACH cluster, identify and describe the common pattern/characteristic shared by its values
2. Compare the clusters to understand their differences
3. Provide a concise natural language description for each cluster that captures its key features

The descriptions should:
- Ignore a small part of spelling error, pay attention to pattern error.
- Be clear and specific (e.g., "Numeric values with 'oz' unit suffix" rather than "Values with units")
- Highlight distinguishing features (e.g., "Ends with period" vs "No period")
- Focus on format/structure rather than semantic meaning
- Be suitable for later comparison to identify canonical vs error patterns

Please respond in the following JSON format:
```json
{{
  "cluster_1": {{
    "cluster_id": {clusters_with_samples[0][0] if clusters_with_samples else 0},
    "description": "Natural language description of the pattern"
  }},
  "cluster_2": {{
    "cluster_id": {clusters_with_samples[1][0] if len(clusters_with_samples) > 1 else 0},
    "description": "Natural language description of the pattern"
  }},
  ...
}}
```

IMPORTANT:
- Provide descriptions for ALL {len(clusters_with_samples)} clusters
- Keep descriptions concise (1-2 sentences)
- Focus on observable patterns, not interpretations
- Use the cluster_id from the input

Only respond with the JSON, no additional text.
"""
    return prompt


def pattern_function_generation_prompt(attr_name, cluster_description, sample_values, 
                                       is_error_function=False, canonical_samples=None):
    """
    生成模式匹配函数的提示词
    
    Args:
        attr_name: 属性名称
        cluster_description: 聚类的自然语言描述
        sample_values: 样本值列表（最多10个）
        is_error_function: 是否为错误模式函数（默认False）
        canonical_samples: 正确模式的样本值列表（仅在is_error_function=True时使用）
    
    Returns:
        prompt: 提示词字符串
    """
    samples_str = '\n'.join([f'- "{s}"' for s in sample_values])
    
    if is_error_function and canonical_samples:
        # 错误模式函数：需要区分错误样本和正确样本
        canonical_str = '\n'.join([f'- "{s}"' for s in canonical_samples[:5]])
        
        prompt = f"""You are a data quality expert. Generate a Python function to identify ERROR values for column '{attr_name}'.

**IMPORTANT**: This is an ERROR PATTERN function. It should:
1. Return True for ERROR values (the candidate pattern below)
2. Return False for CORRECT values (the canonical pattern below)

---

ERROR Pattern (should return True):
Description: {cluster_description}
Examples:
{samples_str}

CORRECT Pattern (should return False):
Examples:
{canonical_str}

---

Your task:
Generate a Python function named 'matches_pattern' that:
1. Takes a single parameter 'value' (string)
2. Returns True ONLY if the value matches the ERROR pattern
3. Returns False for CORRECT pattern values
4. Handles edge cases (empty strings, None, etc.)
5. Is self-contained (only use standard library like re, datetime, etc.)

**CRITICAL REQUIREMENTS:**
- The function MUST distinguish between error and correct patterns
- Analyze the key differences between ERROR and CORRECT patterns
- Focus on characteristics that make ERROR values wrong (e.g., trailing period, wrong case, missing data)
- Test mentally: Does it return True for error examples and False for correct examples?
- Be specific enough to avoid false positives, but general enough to catch similar errors
- You can relax constraints to match more error values of the same type, but ensure it does NOT match correct values

**STRATEGY:**
1. First identify what makes the ERROR pattern different from CORRECT pattern
2. Then write logic that captures those differences
3. Ensure the function is neither too strict (missing similar errors) nor too loose (matching correct values)

Return ONLY the Python function code, no explanations or markdown.

Example output format:
def matches_pattern(value):
    value = value.strip()
    # Your validation logic here
    if ***:
    # Return True for ERROR values
        return True  
    # or False based on your logic
    else:
        return False
"""
    else:
        # 标准模式函数（Canonical）
        prompt = f"""You are a data quality expert. Generate a Python function to validate if a value matches the following pattern.

Column: '{attr_name if attr_name else "unknown"}'
Pattern Description: {cluster_description}

Sample values from this pattern:
{samples_str}

Your task:
Generate a Python function named 'matches_pattern' that:
1. Takes a single parameter 'value' (string)
2. Returns True if the value matches this pattern, False otherwise
3. Handles edge cases (empty strings, None, etc.)
4. Is self-contained (only use standard library like re, datetime, etc.)

IMPORTANT:
- Focus on the key characteristics described in the pattern description
- Be neither too strict nor too loose
- Return ONLY the Python function code, no explanations or markdown

Example output format:
def matches_pattern(value):
    if not value or not isinstance(value, str):
        return False
    value = value.strip()
    # Your validation logic here based on the pattern description
    # For example, if pattern is "5-digit numbers":
    # return value.isdigit() and len(value) == 5
    return True
"""
    
    return prompt

