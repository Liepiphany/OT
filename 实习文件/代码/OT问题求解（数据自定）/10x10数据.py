import jax.numpy as jnp
from ott.solvers.linear import sinkhorn
from ott.geometry import geometry
from ott.problems.linear import linear_problem

# 定义成本矩阵和概率分布
C = jnp.array([
    [0.   , 0.423, 0.645, 0.437, 0.891, 0.963, 0.383, 0.791, 0.528, 0.568],
    [0.423, 0.   , 0.925, 0.071, 0.087, 0.020, 0.832, 0.778, 0.870, 0.978],
    [0.645, 0.925, 0.   , 0.461, 0.780, 0.118, 0.639, 0.143, 0.944, 0.521],
    [0.437, 0.071, 0.461, 0.   , 0.264, 0.774, 0.456, 0.568, 0.018, 0.617],
    [0.891, 0.087, 0.780, 0.264, 0.   , 0.612, 0.616, 0.943, 0.681, 0.359],
    [0.963, 0.020, 0.118, 0.774, 0.612, 0.   , 0.697, 0.060, 0.666, 0.670],
    [0.383, 0.832, 0.639, 0.456, 0.616, 0.697, 0.   , 0.210, 0.128, 0.315],
    [0.791, 0.778, 0.143, 0.568, 0.943, 0.060, 0.210, 0.   , 0.363, 0.570],
    [0.528, 0.870, 0.944, 0.018, 0.681, 0.666, 0.128, 0.363, 0.   , 0.438],
    [0.568, 0.978, 0.521, 0.617, 0.359, 0.670, 0.315, 0.570, 0.438, 0.   ]
])
μ = jnp.array([0.05, 0.1, 0.15, 0.1, 0.2, 0.05, 0.08, 0.07, 0.15, 0.05])
ν = jnp.array([0.12, 0.03, 0.17, 0.08, 0.09, 0.13, 0.1, 0.06, 0.14, 0.08])

# 1. 创建几何对象（定义成本矩阵和正则化参数）
geom = geometry.Geometry(cost_matrix=C, epsilon=1e-3)  # epsilon是正则化参数

# 2. 定义线性最优传输问题
prob = linear_problem.LinearProblem(geom, a=μ, b=ν)

# 3. 使用Sinkhorn算法求解最优传输问题
solver = sinkhorn.Sinkhorn()
result = solver(prob)

# 输出结果
print("最优传输矩阵:")
print(result.matrix)

print("\n传输成本:", jnp.sum(result.matrix * C))

#-------------可视化
import matplotlib.pyplot as plt
import numpy as np

matrix= np.array(result.matrix)

plt.figure(figsize=(10, 8))
plt.imshow(matrix, cmap="viridis", interpolation="nearest")

# 添加颜色条
plt.colorbar(label="Value")

# 添加数值标注
#for i in range(matrix.shape[0]):
#    for j in range(matrix.shape[1]):
#        plt.text(j, i, f"{matrix[i, j]:.1e}", ha="center", va="center", color="white")

# 设置坐标轴标签
plt.xticks(range(matrix.shape[1]))
plt.yticks(range(matrix.shape[0]))
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.title("Matrix Heatmap (Matplotlib)")

plt.show()

