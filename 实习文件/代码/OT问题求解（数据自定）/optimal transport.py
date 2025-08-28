import numpy as np

# ---------- 1. 输入 ----------
a = np.array([0.4, 0.6])          # 源分布
b = np.array([0.5, 0.5])          # 目标分布
C = np.array([[0.2, 0.8],
              [0.6, 0.3]])        # 成本矩阵
reg = 1e-1                        # 正则化参数 λ

# ---------- 2. 初始化 ----------
# 计算核矩阵 K = exp(-C / λ)
K = np.exp(-C / reg)

# ---------- 3. 迭代 Sinkhorn ----------
max_iter = 100
u = np.ones_like(a)   # 行缩放因子
v = np.ones_like(b)   # 列缩放因子

for i in range(max_iter):
    # 行归一化
    u = a / (K @ v + 1e-15)
    # 列归一化
    v = b / (K.T @ u + 1e-15)

# ---------- 4. 得到运输计划 ----------
transport_plan = np.diag(u) @ K @ np.diag(v)

print("最优传输计划：")
print(transport_plan)