"""
基于实际数据分析的错误模式验证器 V2 - 最终版
根据 error_analysis_results 中的实际错误模式创建验证函数
"""

from typing import List, Dict, Tuple
import re


class ErrorPatternValidator:
    """基于实际观察的错误模式验证器"""
    
    def __init__(self):
        """初始化验证器，定义所有验证函数"""
        self.validators = {
            # ========== 通用模式（合并后的函数） ==========
            'null_value': self._check_null_value,
            'trailing_dot': self._check_trailing_dot,
            'trailing_ellipsis': self._check_trailing_ellipsis,
            'trailing_asterisk': self._check_trailing_asterisk,
            'contains_percentage': self._check_contains_percentage,
            'contains_double_apostrophe': self._check_contains_double_apostrophe,
            'contains_extra_spaces': self._check_contains_extra_spaces,
            'ends_with_state_code': self._check_ends_with_state_code,
            'is_airport_code': self._check_is_airport_code,
            'has_timezone_info': self._check_has_timezone_info,
            'has_airline_prefix': self._check_has_airline_prefix,
            
            # ========== 特定模式（无法合并的） ==========
            'unicode_suffix': self._check_unicode_suffix,
            'ounce_vs_oz': self._check_ounce_vs_oz,
            'uppercase_oz': self._check_uppercase_oz,
            'ounce_annotation': self._check_ounce_annotation,
            'contains_noon': self._check_contains_noon,
            'contains_date_pattern': self._check_contains_date_pattern,
            'is_not_available': self._check_is_not_available,
            'contains_estimated': self._check_contains_estimated,
            'contains_delayed': self._check_contains_delayed,
            'contains_contact_airline': self._check_contains_contact_airline,
            'percentage_with_x': self._check_percentage_with_x,
            'one_patients': self._check_one_patients,
        }
    
    def validate_error_pattern(self, correct_examples: List[str], 
                              error_examples: List[str],
                              min_match_ratio: float = 0.6) -> Dict:
        """
        验证错误模式是否符合已知的实际错误模式
        
        Args:
            correct_examples: 正确值示例列表
            error_examples: 错误值示例列表
            min_match_ratio: 最小匹配比例
            
        Returns:
            验证结果字典
        """
        if not correct_examples or not error_examples:
            return {
                'valid': False,
                'reason': 'Empty examples',
                'matched_patterns': [],
                'confidence': 0.0
            }
        
        # 对每个验证函数进行检查
        pattern_matches = []
        for pattern_name, validator_func in self.validators.items():
            match_count = 0
            total_pairs = min(len(correct_examples), len(error_examples), 20)
            
            for i in range(total_pairs):
                correct_val = correct_examples[i] if i < len(correct_examples) else correct_examples[0]
                error_val = error_examples[i] if i < len(error_examples) else error_examples[0]
                
                if validator_func(correct_val, error_val):
                    match_count += 1
            
            match_ratio = match_count / total_pairs if total_pairs > 0 else 0
            if match_ratio >= min_match_ratio:
                pattern_matches.append({
                    'name': pattern_name,
                    'match_ratio': match_ratio,
                    'match_count': match_count,
                    'total_checked': total_pairs
                })
        
        # 判断是否有效
        valid = len(pattern_matches) > 0
        confidence = max([p['match_ratio'] for p in pattern_matches]) if pattern_matches else 0.0
        
        return {
            'valid': valid,
            'matched_patterns': pattern_matches,
            'confidence': confidence,
            'reason': f"Matched {len(pattern_matches)} patterns" if valid else "No pattern matched"
        }
    
    # ==================== 通用验证函数（合并后） ====================
    
    def _check_null_value(self, correct: str, error: str) -> bool:
        """统一验证：错误值为 nan/null/empty 等空值"""
        error_s = str(error).strip().lower()
        return error_s in ['nan', 'null', 'empty', 'none', '']
    
    def _check_trailing_dot(self, correct: str, error: str) -> bool:
        """通用：脏值末尾有点，干净值没有
        应用场景：beers.ounces (12.0 oz. vs 12.0 oz)
        """
        return str(error).strip().endswith('.') and not str(correct).strip().endswith('.')
    
    def _check_trailing_ellipsis(self, correct: str, error: str) -> bool:
        """通用：脏值以...结尾，干净值不以此结尾
        应用场景：beers.brewery_name (缩写)
        """
        return str(error).strip().endswith('...') and not str(correct).strip().endswith('...')
    
    def _check_trailing_asterisk(self, correct: str, error: str) -> bool:
        """通用：脏值末尾有-*，干净值没有
        应用场景：tax.city, tax.state, tax.child_exemp (0-*)
        """
        error_s = str(error).strip()
        correct_s = str(correct).strip()
        return error_s.endswith('-*') and not correct_s.endswith('-*')
    
    def _check_contains_percentage(self, correct: str, error: str) -> bool:
        """通用：脏值包含百分号，干净值不包含
        应用场景：beers.abv
        """
        return '%' in str(error).strip() and '%' not in str(correct).strip()
    
    def _check_contains_double_apostrophe(self, correct: str, error: str) -> bool:
        """通用：脏值包含双引号''，干净值不包含
        应用场景：tax.f_name, tax.l_name
        """
        error_s = str(error).strip()
        correct_s = str(correct).strip()
        return "''" in error_s and "''" not in correct_s
    
    def _check_contains_extra_spaces(self, correct: str, error: str) -> bool:
        """通用：脏值包含多余空格（连续两个空格），干净值不包含
        应用场景：hospital.MeasureName
        """
        error_s = str(error).strip()
        correct_s = str(correct).strip()
        return '  ' in error_s and '  ' not in correct_s
    
    def _check_ends_with_state_code(self, correct: str, error: str) -> bool:
        """通用：脏值以州代码结尾，干净值不以州代码结尾
        应用场景：beers.city
        """
        state_codes = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                      'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                      'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                      'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                      'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
        
        correct_s = str(correct).strip()
        error_s = str(error).strip()
        
        # 检查脏值是否以州代码结尾
        error_has_state = any(error_s.endswith(f" {state}") or error_s.endswith(state) 
                             for state in state_codes)
        
        # 检查干净值是否以州代码结尾
        correct_has_state = any(correct_s.endswith(f" {state}") or correct_s.endswith(state) 
                               for state in state_codes)
        
        return error_has_state and not correct_has_state
    
    def _check_is_airport_code(self, correct: str, error: str) -> bool:
        """通用：脏值是机场代码（3个大写字母），干净值不是
        应用场景：flights.Destination_Airport, flights.Origin_Airport
        """
        error_s = str(error).strip()
        correct_s = str(correct).strip()
        is_code_error = bool(re.match(r'^[A-Z]{3}$', error_s))
        is_code_correct = bool(re.match(r'^[A-Z]{3}$', correct_s))
        return is_code_error and not is_code_correct
    
    def _check_has_timezone_info(self, correct: str, error: str) -> bool:
        """通用：脏值包含时区信息（如 +00:00），干净值不包含
        应用场景：flights.Scheduled_Departure, flights.Scheduled_Arrival, flights.Actual_Arrival
        """
        error_s = str(error).strip()
        correct_s = str(correct).strip()
        has_tz_error = bool(re.search(r'[+-]\d{2}:\d{2}', error_s))
        has_tz_correct = bool(re.search(r'[+-]\d{2}:\d{2}', correct_s))
        return has_tz_error and not has_tz_correct
    
    def _check_has_airline_prefix(self, correct: str, error: str) -> bool:
        """通用：脏值有航空公司前缀（如 AA123），干净值没有
        应用场景：flights.Flight_Number
        """
        error_s = str(error).strip()
        correct_s = str(correct).strip()
        has_prefix_error = bool(re.match(r'^[A-Z]{2}\d+', error_s))
        has_prefix_correct = bool(re.match(r'^[A-Z]{2}\d+', correct_s))
        return has_prefix_error and not has_prefix_correct
    
    # ==================== 特定验证函数（无法合并） ====================
    
    def _check_unicode_suffix(self, correct: str, error: str) -> bool:
        """特定：脏值以 �? 结尾，干净值不以此结尾
        应用场景：beers.beer_name
        """
        return str(error).endswith('�?') and not str(correct).endswith('�?')
    
    def _check_ounce_vs_oz(self, correct: str, error: str) -> bool:
        """特定：脏值包含ounce，干净值不包含
        应用场景：beers.ounces (oz vs ounce)
        """
        correct_s = str(correct).strip().lower()
        error_s = str(error).strip().lower()
        return 'ounce' in error_s and 'ounce' not in correct_s
    
    def _check_uppercase_oz(self, correct: str, error: str) -> bool:
        """特定：脏值包含OZ大写，干净值不包含
        应用场景：beers.ounces
        """
        correct_s = str(correct).strip()
        error_s = str(error).strip()
        return 'OZ' in error_s and 'OZ' not in correct_s
    
    def _check_ounce_annotation(self, correct: str, error: str) -> bool:
        """特定：脏值有额外注释，干净值没有
        应用场景：beers.ounces (Alumi-Tek, Silo Can)
        """
        correct_s = str(correct).strip()
        error_s = str(error).strip()
        keywords = ['Alumi', 'Silo', 'Can']
        has_in_error = any(word in error_s for word in keywords)
        has_in_correct = any(word in correct_s for word in keywords)
        return has_in_error and not has_in_correct
    
    def _check_contains_noon(self, correct: str, error: str) -> bool:
        """特定：脏值包含 noon，干净值不包含
        应用场景：flights 时间格式
        """
        error_s = str(error).strip().lower()
        correct_s = str(correct).strip().lower()
        return 'noon' in error_s and 'noon' not in correct_s
    
    def _check_contains_date_pattern(self, correct: str, error: str) -> bool:
        """特定：脏值包含日期模式，干净值不包含
        应用场景：flights 时间+日期
        """
        error_s = str(error).strip()
        correct_s = str(correct).strip()
        date_patterns = [r'Dec', r'aDec', r'\d{1,2}/\d{1,2}', r'Fri', r'Thu']
        has_in_error = any(re.search(p, error_s) for p in date_patterns)
        has_in_correct = any(re.search(p, correct_s) for p in date_patterns)
        return has_in_error and not has_in_correct
    
    def _check_is_not_available(self, correct: str, error: str) -> bool:
        """特定：脏值是 Not Available，干净值不是
        应用场景：flights 占位符
        """
        return str(error).strip() == 'Not Available' and str(correct).strip() != 'Not Available'
    
    def _check_contains_estimated(self, correct: str, error: str) -> bool:
        """特定：脏值包含 Estimated 或 runway，干净值不包含
        应用场景：flights 状态标记
        """
        error_s = str(error).strip().lower()
        correct_s = str(correct).strip().lower()
        has_in_error = 'estimated' in error_s or 'runway' in error_s
        has_in_correct = 'estimated' in correct_s or 'runway' in correct_s
        return has_in_error and not has_in_correct
    
    def _check_contains_delayed(self, correct: str, error: str) -> bool:
        """特定：脏值包含 Delayed，干净值不包含
        应用场景：flights 状态标记
        """
        error_s = str(error).strip()
        correct_s = str(correct).strip()
        return 'Delayed' in error_s and 'Delayed' not in correct_s
    
    def _check_contains_contact_airline(self, correct: str, error: str) -> bool:
        """特定：脏值包含 Contact Airline，干净值不包含
        应用场景：flights 占位符
        """
        error_s = str(error).strip()
        correct_s = str(correct).strip()
        return 'Contact Airline' in error_s and 'Contact Airline' not in correct_s
    
    def _check_percentage_with_x(self, correct: str, error: str) -> bool:
        """特定：脏值百分号中包含x，干净值不包含
        应用场景：hospital.Score
        """
        error_s = str(error).strip().lower()
        correct_s = str(correct).strip().lower()
        has_x_error = '%' in error_s and 'x' in error_s
        has_x_correct = '%' in correct_s and 'x' in correct_s
        return has_x_error and not has_x_correct
    
    def _check_one_patients(self, correct: str, error: str) -> bool:
        """特定：脏值是 1 patients（单数用复数），干净值不是
        应用场景：hospital.Sample
        """
        error_s = str(error).strip().lower()
        correct_s = str(correct).strip().lower()
        return error_s == '1 patients' and correct_s != '1 patients'


if __name__ == '__main__':
    # 测试
    validator = ErrorPatternValidator()
    
    print("=" * 60)
    print("测试验证器")
    print("=" * 60)
    
    # 测试1: null值
    print("\n测试1: null值统一验证")
    result = validator.validate_error_pattern(
        ['nan (wait to be cleaned)', 'value'],
        ['nan', 'null']
    )
    print(f"  有效: {result['valid']}, 置信度: {result['confidence']:.1%}")
    
    # 测试2: Hospital - 字符替换
    print("\n测试2: Hospital ProviderNumber 字符替换")
    result = validator.validate_error_pattern(
        ['10024', '10029'],
        ['1xx24', '1xx29']
    )
    print(f"  有效: {result['valid']}, 置信度: {result['confidence']:.1%}")
    
    # 测试3: Tax - 双引号
    print("\n测试3: Tax name 双引号")
    result = validator.validate_error_pattern(
        ["Jun'ichi", "Ken'ichi"],
        ["Jun''ichi", "Ken''ichi"]
    )
    print(f"  有效: {result['valid']}, 置信度: {result['confidence']:.1%}")
    
    print("\n测试完成！")

