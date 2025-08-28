if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(version = "3.21")

BiocManager::install(c("splatter", "scater"))
library(splatter)
library(scater)
# R 代码 - 生成基础数据 X
library(splatter)

# 设置参数（示例值，需根据 D.2 调整）
params <- newSplatParams(
  nGenes = 300,       # 基因数 l
  batchCells = 1000,     # 细胞数 n1
  mean.shape = 0.6,     # Gamma 分布参数
  mean.rate = 0.3,      # Gamma 分布参数
  lib.loc = 11,         # 文库大小
  out.prob = 0.05,      # 零膨胀概率
  dropout.type = "experiment"
)

# 生成数据
sim <- splatSimulate(params, method = "groups")
X <- counts(sim)  # 原始计数矩阵 [n1 x l]
write.csv(X,"C:\\Users\\lizhenwang\\Desktop\\base_data.csv")
write.table(X, file="C:\\Users\\lizhenwang\\Desktop\\base_data.txt", 
            sep="\t", row.names=FALSE, col.names=FALSE)
