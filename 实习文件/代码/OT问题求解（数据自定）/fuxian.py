import jax.numpy as jnp
from ott.solvers.linear import sinkhorn
from ott.geometry import geometry
from ott.problems.linear import linear_problem

# 定义成本矩阵和概率分布
C = jnp.array([[0., 0.1, 0.3],
               [0.1, 0., 0.6],
               [0.3, 0.6, 0.]])
μ = jnp.array([0.46,0.28,0.26])
ν = jnp.array([0.33,0.27,0.4])

# 1. 创建几何对象（定义成本矩阵和正则化参数）
geom = geometry.Geometry(cost_matrix=C, epsilon=1e-3)  # epsilon是正则化参数

print("geom:",geom)

# 2. 定义线性最优传输问题
prob = linear_problem.LinearProblem(geom, a=μ, b=ν)

print("prob:",prob)

# 3. 使用Sinkhorn算法求解最优传输问题
solver = sinkhorn.Sinkhorn()
result = solver(prob)

# 输出结果
print("最优传输矩阵:")
print(result.matrix)

print("\n传输成本:", jnp.sum(result.matrix * C))