import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # 使用非交互式后端，便于脚本环境直接保存图片

# 加载 DDIM 生成的样本数据
samples = np.load("ddim_samples.npy")

# 将前 16 张样本按 4x4 网格排布并保存为 PNG
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(samples[i * 4 + j].squeeze(), cmap="gray")
        axs[i, j].axis("off")

plt.savefig("ddim_samples.png", bbox_inches="tight")
plt.close()
print("已保存为 ddim_samples.png")
