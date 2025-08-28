import pathlib, numpy as np, seaborn as sns, matplotlib.pyplot as plt

out_dir = pathlib.Path("C:\\Users\\lizhenwang\\Desktop\\test_transport_matrices")
heat_dir = out_dir / "heatmaps"
heat_dir.mkdir(exist_ok=True)

for p in sorted(out_dir.glob("T_*.npy")):
    T = np.load(p)
    dose = p.stem.split('_')[1]
    plt.figure(figsize=(3,2.5))
    sns.heatmap(T, cmap="viridis", cbar=False)
    plt.title(f"dose={dose}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(heat_dir / f"{p.stem}.png", dpi=150)
    plt.close()