import matplotlib.pyplot as plt
import numpy as np

num1 = [1.4,3.3,2.1,3.4,2.5,3.7,4.1,3.9,9.2,10,10.2,9.1,7.3,6.5,7.7,7.8,8.1,2.9,3.5,10,10,9.3,8.2,9.1,2.4,7.6,3.1,8.4,6.9,7.4]
num2 = [89,85,87,87,89,92,93,95,18,13,18,19,54,56,72,49,31,81,90,22,17,22,13,24,75,26,97,68,69,59]
cluster = 3

# 把 1×30 的二维数组拉成 30×1，再横向拼成 30×2 的坐标
data = np.column_stack((num1, num2))


#随机选取初始质心
def center_point(data, cluster):
    """从data个数据中,返回 cluster 个初始中心，这里用随机选点的方式"""
    indices = np.random.choice(len(data), size=cluster, replace=False)#a：一维数组或 int size：输出形状（int 或 tuple）；不给则返回单个标量 replace=True/False：是否放回抽样
    return data[indices].astype(float)

# ---------------- K-means 主流程 ----------------
def kmeans(data, k, max_iter=100, tol=1.0e-9):#max_iter：最大迭代轮数，防止死循环。tol：质心移动距离小于该阈值就视为收敛，提前结束。
    centers = center_point(data, k)          # 1. 初始化中心
    for _ in range(max_iter):
        # 2. 分配样本到最近的中心
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)#不懂 for i in(len(data)):(data[i,0]-center[j,0])^2+(data[i,1]-center[j,1])^2
        labels = np.argmin(distances, axis=1)

        # 3. 计算新中心
        new_centers = np.array([data[labels == i].mean(axis=0) if np.any(labels == i)
                                else centers[i]            # 若某类为空，保留原中心
                                for i in range(k)])

        # 4. 收敛判断
        if np.all(np.linalg.norm(new_centers - centers, axis=1) < tol):
            break
        centers = new_centers
    print("实际迭代轮数：", _+1) 
    return centers, labels
    

centers, labels = kmeans(data, cluster)
# ---------------- 画图 ----------------
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
plt.title('K-means clustering (k=3)')
plt.xlabel('num1')
plt.ylabel('num2')
plt.show()

values, counts = np.unique(labels, return_counts=True)

print(values)   # [1 2 3 5]
print(counts /30)  
print(centers)