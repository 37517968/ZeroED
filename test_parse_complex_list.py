#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 parse_complex_list 函数的修复效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))



def test_extract_from_file():
    """测试从文件中提取 enhanced info"""
    print("\n=== 测试从文件中提取 enhanced info ===")
    
    file_path = "./result/pipeline/11-09 flights01-5%-set1-iterations3/enhanced/clean_gen_flight.txt"
    
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
        
        print(f"文件路径: {file_path}")
        print(f"文件内容长度: {len(file_content)} 字符")
        
        # 使用 extract_enhanced_info 函数提取信息
        from main import extract_enhanced_info
        result = extract_enhanced_info(file_content, 'flight')
        
        print(f"\n提取到的信息数量: {len(result)}")
        
        # 显示前5条提取的信息
        print("\n前5条提取的信息:")
        for i, info in enumerate(result[:5]):
            print(f"{i+1}. {info}")
        
        # 统计不同类型的信息
        flight_values = []
        for info in result:
            if len(info) >= 4 and isinstance(info[3], dict):
                flight_values.append(info[3].get('flight', 'N/A'))
        
        print(f"\n提取到的航班值数量: {len(flight_values)}")
        print("前10个航班值:")
        for i, flight in enumerate(flight_values[:10]):
            print(f"{i+1}. {flight}")
        
        print("✅ 文件提取测试成功")
        
    except FileNotFoundError:
        print(f"❌ 文件未找到: {file_path}")
    except Exception as e:
        print(f"❌ 文件提取测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extract_from_file()
    print("\n=== 测试完成 ===")