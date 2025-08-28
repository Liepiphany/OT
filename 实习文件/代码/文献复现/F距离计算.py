import numpy as np
import pathlib
from glob import glob
import matplotlib.pyplot as plt

# 设置路径
out_dir_T = pathlib.Path("C:\\Users\\lizhenwang\\Desktop\\rewrite_transport_matrices")

# 加载基线运输矩阵
T_baseline = np.load(out_dir_T / "T_baseline.npy")

# 计算相对Frobenius误差
def relative_frobenius_error(pred, true):
    return np.linalg.norm(pred - true, 'fro') / np.linalg.norm(true, 'fro')

# 遍历所有剂量的真实T
errors = []
doses=[]
for fp in sorted(glob(str(out_dir_T / "T_*.npy"))):
    if "baseline" in fp:  # 跳过基线文件
        continue
    T_true = np.load(fp)
    err = relative_frobenius_error(T_baseline, T_true)
    dose = fp.split('_')[-1].replace('.npy', '')
    print(f"Dose {dose}: F-distance = {err:.4f}")
    errors.append(err)
    doses.append(dose)

# 平均误差
print(f"\n平均F-distance（基线）= {np.mean(errors):.4f}")

plt.figure(figsize=(6, 4))
plt.plot(doses, errors, marker='o', linestyle='-', color='tab:blue')
plt.xlabel("Dosage")
plt.ylabel("Relative Frobenius Error")
plt.title("Baseline Error vs Dosage")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()