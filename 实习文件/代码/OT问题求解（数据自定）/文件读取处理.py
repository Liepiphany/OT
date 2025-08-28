import jax.numpy as jnp
import pandas as pd
from ott.solvers.linear import sinkhorn
from ott.geometry import geometry
from ott.problems.linear import linear_problem

# 读取数据（假设 CSV 文件）
df = pd.read_csv(r"C:\Users\lizhenwang\Desktop\data.csv")

# 定义成本矩阵（假设是固定的）
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
    [0.568, 0.978, 0.521, 0.617, 0.359, 0.670, 0.315, 0.570, 0.438, 0.   ]])

# 正则化参数
epsilon = 1e-3

# 存储结果
results = []

# 遍历每一行（每一对 μ 和 ν）
for _, row in df.iterrows():
    μ = jnp.array([row["mu1"], row["mu2"], row["mu3"],row["mu4"],row["mu5"], row["mu6"], row["mu7"],row["mu8"],row["mu9"],row["mu10"]])
    ν = jnp.array([row["v1"], row["v2"], row["v3"], row["v4"], row["v5"], row["v6"], row["v7"], row["v8"], row["v9"], row["v10"]])
    
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


#输出结果（示例）
for res in results:
   #print(f"μ: {res['mu']}, ν: {res['nu']}")
   #print("Transport Matrix:")
   print(res["transport_matrix"])
   #print(f"Cost: {res['cost']}\n")