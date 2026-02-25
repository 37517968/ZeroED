"""
乱码字符检测模块
用于在数据清洗流程中直接识别包含乱码字符的数据
"""

import re
from typing import Union, List, Dict


class GarbledCharDetector:
    """
    乱码字符检测器
    
    检测以下类型的乱码：
    1. Unicode组合字符（如 ̩ ̱ ̦ ̴ ̬ 等）
    2. 替换字符 � (U+FFFD)
    3. 控制字符和非打印字符
    4. 异常的Unicode范围字符
    5. 常见的编码错误模式
    """
    
    def __init__(self):
        """初始化检测器，定义乱码模式"""
        # 基本乱码模式
        self.garbled_patterns = [
            (r'�', '替换字符'),  # Unicode替换字符
            (r'[\u0300-\u036F]', 'Unicode组合变音符号'),  # 组合变音符号
            (r'[\u0000-\u0008\u000B-\u000C\u000E-\u001F]', '控制字符'),  # 控制字符（排除\t\n\r）
            (r'[\uFFF0-\uFFFF]', '特殊用途字符'),  # 特殊用途字符
            (r'[\u200B-\u200F]', '零宽字符'),  # 零宽字符
            (r'[\u2028-\u202F]', '行段分隔符'),  # 行/段分隔符
            (r'[\uFFFD]', 'Unicode替换字符'),  # 明确的替换字符
        ]
        
        # 编译正则表达式以提高性能
        self.compiled_patterns = [
            (re.compile(pattern), name) 
            for pattern, name in self.garbled_patterns
        ]
        
        # 允许的基本字符集（ASCII + 常见扩展拉丁字符 + 常见标点）
        # 这个模式匹配"正常"字符，不在此范围内的可能是乱码
        self.normal_char_pattern = re.compile(
            r'^[\x20-\x7E'  # 基本ASCII可打印字符
            r'\u00A0-\u00FF'  # 拉丁扩展-A（包括常见的重音字符）
            r'\u0100-\u017F'  # 拉丁扩展-B
            r'\u0180-\u024F'  # 拉丁扩展-C和D
            r'\u2000-\u206F'  # 常规标点符号
            r'\u3000-\u303F'  # CJK符号和标点
            r'\u4E00-\u9FFF'  # CJK统一表意文字
            r'\uAC00-\uD7AF'  # 韩文音节
            r'\u0400-\u04FF'  # 西里尔字母
            r'\u0370-\u03FF'  # 希腊字母
            r'\t\n\r'  # 允许的控制字符
            r']+$'
        )
    
    def contains_garbled_chars(self, text: Union[str, None]) -> bool:
        """
        检测文本是否包含乱码字符
        
        Args:
            text: 要检测的文本
            
        Returns:
            bool: 如果包含乱码字符返回True，否则返回False
        """
        if text is None or text == '':
            return False
        
        text_str = str(text)
        
        # 检查是否匹配任何乱码模式
        for pattern, _ in self.compiled_patterns:
            if pattern.search(text_str):
                return True
        
        return False
    
    def detect_garbled_details(self, text: Union[str, None]) -> Dict:
        """
        检测文本中的乱码字符并返回详细信息
        
        Args:
            text: 要检测的文本
            
        Returns:
            dict: 包含检测结果的字典
                - has_garbled: bool, 是否包含乱码
                - garbled_types: list, 检测到的乱码类型
                - garbled_chars: list, 检测到的乱码字符及其位置
                - clean_text: str, 移除乱码后的文本（仅供参考）
        """
        if text is None or text == '':
            return {
                'has_garbled': False,
                'garbled_types': [],
                'garbled_chars': [],
                'clean_text': text
            }
        
        text_str = str(text)
        garbled_types = []
        garbled_chars = []
        
        # 检查每种乱码模式
        for pattern, name in self.compiled_patterns:
            matches = pattern.finditer(text_str)
            for match in matches:
                garbled_types.append(name)
                garbled_chars.append({
                    'char': match.group(),
                    'position': match.start(),
                    'type': name,
                    'unicode': f'U+{ord(match.group()):04X}'
                })
        
        # 生成清理后的文本（移除所有乱码字符）
        clean_text = text_str
        for pattern, _ in self.compiled_patterns:
            clean_text = pattern.sub('', clean_text)
        
        return {
            'has_garbled': len(garbled_chars) > 0,
            'garbled_types': list(set(garbled_types)),
            'garbled_chars': garbled_chars,
            'clean_text': clean_text
        }
    
    def batch_detect(self, texts: List[str]) -> List[bool]:
        """
        批量检测多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            list: 布尔值列表，对应每个文本是否包含乱码
        """
        return [self.contains_garbled_chars(text) for text in texts]
    
    def filter_garbled(self, texts: List[str]) -> List[str]:
        """
        过滤掉包含乱码的文本
        
        Args:
            texts: 文本列表
            
        Returns:
            list: 不包含乱码的文本列表
        """
        return [text for text in texts if not self.contains_garbled_chars(text)]
    
    def get_garbled_indices(self, texts: List[str]) -> List[int]:
        """
        获取包含乱码的文本的索引
        
        Args:
            texts: 文本列表
            
        Returns:
            list: 包含乱码的文本的索引列表
        """
        return [i for i, text in enumerate(texts) if self.contains_garbled_chars(text)]


def is_garbled(text: Union[str, None]) -> bool:
    """
    便捷函数：快速检测文本是否包含乱码
    
    Args:
        text: 要检测的文本
        
    Returns:
        bool: 如果包含乱码返回True，否则返回False
    """
    detector = GarbledCharDetector()
    return detector.contains_garbled_chars(text)


if __name__ == '__main__':
    # 测试
    import sys
    # 设置输出编码为UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    detector = GarbledCharDetector()
    
    print("=" * 80)
    print("乱码字符检测器测试")
    print("=" * 80)
    
    # 测试用例
    test_cases = [
        ("正常文本", "Jrgen Rehm", False),
        ("替换字符", "J�_rgen Rehm", True),
        ("组合字符1", "P G̩rard", True),
        ("组合字符2", "Benjam�_n Pi̱a", True),
        ("组合字符3", "Livia G�_mez", True),
        ("多个乱码", "Jara P̩rez-Jim̩nez", True),
        ("正常中文", "中文测试", False),
        ("正常英文", "Normal English Text", False),
        ("正常重音", "café résumé naïve", False),
        ("正常符号", "test@example.com (123) 456-7890", False),
        ("零宽字符", "test\u200Btext", True),
        ("控制字符", "test\x00text", True),
    ]
    
    print("\n基本检测测试:")
    print("-" * 80)
    for name, text, expected in test_cases:
        result = detector.contains_garbled_chars(text)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"{status} {name:20s} | 期望: {expected:5} | 结果: {result:5} | 文本: {text[:40]}")
    
    print("\n\n详细检测测试:")
    print("-" * 80)
    garbled_examples = [
        "J�_rgen Rehm",
        "P G̩rard",
        "Benjam�_n Pi̱a",
        "Henri��tte A Smit",
    ]
    
    for text in garbled_examples:
        details = detector.detect_garbled_details(text)
        print(f"\n原文: {text}")
        print(f"  包含乱码: {details['has_garbled']}")
        print(f"  乱码类型: {details['garbled_types']}")
        print(f"  乱码字符数: {len(details['garbled_chars'])}")
        if details['garbled_chars']:
            for char_info in details['garbled_chars'][:3]:  # 只显示前3个
                print(f"    - 位置 {char_info['position']}: '{char_info['char']}' ({char_info['unicode']}) - {char_info['type']}")
        print(f"  清理后: {details['clean_text']}")
    
    print("\n\n批量检测测试:")
    print("-" * 80)
    batch_texts = [
        "Normal text 1",
        "J�_rgen Rehm",
        "Another normal text",
        "P G̩rard",
        "Clean text"
    ]
    results = detector.batch_detect(batch_texts)
    garbled_indices = detector.get_garbled_indices(batch_texts)
    
    print(f"总文本数: {len(batch_texts)}")
    print(f"包含乱码的文本数: {sum(results)}")
    print(f"包含乱码的索引: {garbled_indices}")
    
    print("\n过滤后的干净文本:")
    clean_texts = detector.filter_garbled(batch_texts)
    for i, text in enumerate(clean_texts, 1):
        print(f"  {i}. {text}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

