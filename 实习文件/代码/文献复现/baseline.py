from glob import glob
import numpy as np
import pandas as pd
from ot import sinkhorn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pathlib

# ---------- 路径 ----------
data_dir  = pathlib.Path("C:\\Users\\lizhenwang\\Desktop\\re_perturbation_data")          # 50 份扰动后的 dose_XXX.csv
ctrl_path = pathlib.Path("C:\\Users\\lizhenwang\\Desktop\\base_data.csv")  # 真正的对照（未扰动）
out_dir_T = pathlib.Path("C:\\Users\\lizhenwang\\Desktop\\repro_transport_matrices")
out_dir_T.mkdir(exist_ok=True)

n_clusters = 8


# ---------- 1. 计算固定 μ ----------
Y_ctrl = pd.read_csv(ctrl_path, header=None).values.T        # 转置：现在形状为 (300, 1000)
clusterer = KMeans(n_clusters=n_clusters, random_state=0)
labels_ctrl = clusterer.fit_predict(Y_ctrl)                  # 对特征（列）进行聚类
μ = np.bincount(labels_ctrl, minlength=n_clusters).astype(float)
μ /= μ.sum()


# ---------- 2. 计算成本矩阵 C ----------
centers = clusterer.cluster_centers_        # 8×1000
C = euclidean_distances(centers, centers)
C /= C.max() + 1e-8


# 读取所有剂量文件
dose_files = sorted(glob(str(data_dir / "dose_*.csv")))

nus = []
for f in dose_files:
    Y = pd.read_csv(f, header=None).values.T            # (300, n_cells)
    labels = clusterer.predict(Y)                       # 用同一个 clusterer
    nu = np.bincount(labels, minlength=n_clusters).astype(float)
    nu /= nu.sum()
    nus.append(nu)

# 平均目标分布
nu_avg = np.mean(nus, axis=0)

T_baseline = sinkhorn(μ, nu_avg, C, reg=0.001,
             numItermax=1000000,   # 原文没给，但 10 万步通常够用
             stopThr=1e-6)
np.save(out_dir_T / "T_baseline.npy", T_baseline)