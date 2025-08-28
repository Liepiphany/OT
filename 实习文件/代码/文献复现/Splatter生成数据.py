# Python中调用R生成数据
import pandas as pd
import numpy as np
from scipy.stats import gamma, poisson

# 读取基础数据
base_data_path = "C:\\Users\\lizhenwang\\Desktop\\base_data.csv"
X = np.loadtxt("C:\\Users\\lizhenwang\\Desktop\\base_data.txt", delimiter="\t")

np.random.seed(42)
n_cells, n_genes = X.shape
responsive_genes = np.random.choice(n_genes, size=int(0.9*n_genes), replace=False)
unresponsive_cells = np.random.choice(n_cells, size=int(0.15*n_cells), replace=False)
# 扰动数据生成 (Python)

def apply_perturbation(Y_bar, p_i, effect_type):
    Y_pert = Y_bar.copy().astype(float)
    for gene_idx in responsive_genes:
        amplitude = np.random.uniform(0.3, 1)
        for cell_idx in range(n_cells):
            if cell_idx in unresponsive_cells: continue
            expr = Y_bar[cell_idx, gene_idx]
            if effect_type == "reciprocal":
                a,b=3.0,1.0
                perturbed = a * expr + b
            else:  # reciprocal_root
                a,b=100,0.2
                perturbed = a * (expr + 1e-6) ** (-b)
            Y_pert[cell_idx, gene_idx] = expr + amplitude * (perturbed - expr)
    return Y_pert

# 生成完整数据集
dosages = np.linspace(0, 1, 50)  # 50个剂量点
effect_type = "linear"
dataset = [(X, apply_perturbation(X, p_i, effect_type), p_i) for p_i in dosages]

import os
import pandas as pd
import numpy as np

# 创建输出目录
output_dir = "C:\\Users\\lizhenwang\\Desktop\\test_perturbation_data"
os.makedirs(output_dir, exist_ok=True)

# 导出每个剂量点的数据
for i, (base_data, pert_data, dosage) in enumerate(dataset):
    # 创建DataFrame
    df = pd.DataFrame(pert_data)
    
    # 添加行和列标签（如果有的话）
    # 如果没有基因/细胞名称，可以使用默认索引
    # 如果有名称，可以这样添加：
    # df.columns = gene_names
    # df.index = cell_names
    

    
    # 设置文件名（使用剂量值，但替换小数点以避免文件系统问题）
    filename = f"dose_{dosage:.3f}.csv".replace('.', '_')
    filepath = os.path.join(output_dir, filename)
    
    # 保存为CSV
    df.to_csv(filepath, index=False,header=False)
    
    print(f"已保存: {filename}")

# 创建元数据文件（记录响应基因信息）
metadata = {
    "responsive_genes": responsive_genes.tolist(),
    "unresponsive_cells": unresponsive_cells.tolist(),
    "effect_type": effect_type,
    "n_cells": n_cells,
    "n_genes": n_genes
}

metadata_df = pd.DataFrame([metadata])
metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False,header=None)

print("所有文件已保存完成!")
print(f"共生成 {len(dataset)} 个剂量点数据文件")
print(f"输出目录: {os.path.abspath(output_dir)}")
