#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_csv_rows.py
将目录下所有无表头 CSV 的每一行除以该行均值
优化版本
"""

import os
import glob
import pandas as pd
import numpy as np

# ========= 用户可修改区域 =========
INPUT_DIR   = 'C:\\Users\\lizhenwang\\Desktop\\perturbation_data'  # 原始 CSV 所在目录
OUTPUT_DIR  = 'C:\\Users\\lizhenwang\\Desktop\\new_perturbation_data'  # 处理后 CSV 输出目录
CSV_PATTERN = '*.csv'              # 文件通配符
PREFIX      = 'c'                  # 新文件前缀
# =================================

def divide_by_row_mean(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 DataFrame 逐行处理：每行数值除以该行均值。
    非数值单元格转换为 NaN；行均值为 0 的行全部设为 0。
    """
    # 复制一份，将所有数据转换为数值类型，非数值变为NaN
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    
    # 计算每行的平均值
    row_means = numeric_df.mean(axis=1)
    
    # 处理除零情况：将均值为0的行设为NaN，避免除零错误
    row_means[row_means == 0] = np.nan
    
    # 将每行数据除以该行的平均值
    result = numeric_df.div(row_means, axis=0)
    
    # 将NaN值替换为0（处理除零情况和原始非数值情况）
    result = result.fillna(0)
    
    return result

def process_all_files():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有匹配的CSV文件
    search_path = os.path.join(INPUT_DIR, CSV_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f'未找到匹配文件：{search_path}')
        return
    
    # 处理每个文件
    for fp in files:
        fname = os.path.basename(fp)
        out_name = PREFIX + fname
        out_path = os.path.join(OUTPUT_DIR, out_name)
        
        try:
            # 无表头 CSV → header=None
            df = pd.read_csv(fp, header=None)
            df_norm = divide_by_row_mean(df)
            df_norm.to_csv(out_path, header=False, index=False)
            print(f'已生成 → {out_path}')
        except Exception as e:
            print(f'处理文件 {fp} 时出错: {str(e)}')

if __name__ == '__main__':
    process_all_files()