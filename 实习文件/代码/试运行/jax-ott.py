# --------------------------------------------------
# 1. 环境准备
# --------------------------------------------------
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ott.geometry import geometry   # 自定义成本矩阵时用这个
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# --------------------------------------------------
# 2. 自己设计数据
# --------------------------------------------------
# 2.1 点坐标
n, m, d = 5, 7, 2
x = jnp.array([[0., 0.], [1., 1.], [2., 0.], [1., -1.], [0., -2.]])   # (n, d)
y = jnp.array([[0., 1.], [1., 2.], [2., 1.], [3., 1.],
               [3., 0.], [2., -1.], [1., -2.]])                        # (m, d)

# 2.2 权重（概率）
a = jnp.array([0.1, 0.2, 0.3, 0.2, 0.2])        # 和为 1
b = jnp.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1])  # 和为 1

# 2.3 自定义成本矩阵（例如：欧氏距离，而非距离平方）
#     这里我们手动构造，也可以写自己的公式
C = jax.vmap(lambda xi: jax.vmap(lambda yj: jnp.linalg.norm(xi - yj))(y))(x)  # (n, m)

# --------------------------------------------------
# 3. 构建 OT 问题并求解
# --------------------------------------------------
geom  = geometry.Geometry(cost_matrix=C, epsilon=0.05)   # epsilon 就是正则化强度 ε
prob  = linear_problem.LinearProblem(geom, a, b)
solver = sinkhorn.Sinkhorn()
out = solver(prob)

# --------------------------------------------------
# 4. 取出运输矩阵 & 可视化
# --------------------------------------------------
transport = out.matrix   # (n, m)

print("成本矩阵 C (n×m):\n", C)
print("\n运输矩阵 γ (n×m):\n", transport)

# 4.1 可视化
plt.figure(figsize=(6, 5))
plt.title("Transport matrix γ")
plt.imshow(transport, cmap="Blues")
plt.colorbar(label="mass transported")
plt.xticks(range(m), [f"y{j}" for j in range(m)])
plt.yticks(range(n), [f"x{i}" for i in range(n)])
plt.xlabel("target points y")
plt.ylabel("source points x")
plt.tight_layout()
plt.show()