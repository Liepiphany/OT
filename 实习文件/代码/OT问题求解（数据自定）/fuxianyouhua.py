import jax.numpy as jnp
import pandas as pd
from ott.solvers.linear import sinkhorn
from ott.geometry import geometry
from ott.problems.linear import linear_problem

# 读取数据（假设 CSV 文件）
df = pd.read_csv(r"C:\Users\lizhenwang\Desktop\test.csv")

# 定义成本矩阵（假设是固定的）
C = jnp.array([[0., 0.1, 0.3],
               [0.1, 0., 0.6],
               [0.3, 0.6, 0.]])

# 正则化参数
epsilon = 1e-3

# 存储结果
results = []

# 遍历每一行（每一对 μ 和 ν）
for _, row in df.iterrows():
    μ = jnp.array([row["mu1"], row["mu2"], row["mu3"]])
    ν = jnp.array([row["v1"], row["v2"], row["v3"]])
    
    # 计算最优传输
    geom = geometry.Geometry(cost_matrix=C, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a=μ, b=ν)
    solver = sinkhorn.Sinkhorn()
    result = solver(prob)
    
    # 计算传输成本
    cost = jnp.sum(result.matrix * C)
    
    # 存储结果
    results.append({
        "mu": μ,
        "nu": ν,
        "transport_matrix": result.matrix,
        "cost": cost,
    })


# 输出结果（示例）
for res in results:
    print(f"μ: {res['mu']}, ν: {res['nu']}")
    print("Transport Matrix:")
    print(res["transport_matrix"])
    print(f"Cost: {res['cost']}\n")