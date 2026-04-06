import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

matplotlib.use("Agg")  # 使用非交互式后端，便于脚本直接保存图片

# ===================== 1. 设备与超参数配置 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 扩散过程参数，需要与训练阶段保持一致
T = 1000  # 训练时总扩散步数
img_size = 28
img_channels = 1

# 线性噪声调度
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t，累计保留的信号比例
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)


# ===================== 2. U-Net 噪声预测模型 =====================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # 用正余弦函数把时间步编码成向量，让模型感知当前去噪阶段
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            # 上采样阶段会先拼接跳跃连接，因此输入通道数翻倍
            self.conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, time_emb):
        # 先提取空间特征，再把时间嵌入加到每个通道上
        h = self.relu(self.norm1(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(time_emb))
        h = h + time_emb[(...,) + (None,) * 2]
        h = self.relu(self.norm2(self.conv2(h)))
        return self.transform(h)


class UNet(nn.Module):
    def __init__(self, image_channels=1, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        self.init_conv = nn.Conv2d(image_channels, 64, 3, padding=1)
        self.downs = nn.ModuleList(
            [
                Block(64, 64, time_emb_dim),
                Block(64, 128, time_emb_dim),
                Block(128, 256, time_emb_dim),
            ]
        )
        self.ups = nn.ModuleList(
            [
                Block(256, 128, time_emb_dim, up=True),
                Block(128, 64, time_emb_dim, up=True),
                Block(64, 64, time_emb_dim, up=True),
            ]
        )
        self.final_conv = nn.Conv2d(64, image_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.init_conv(x)
        residual_connections = []

        # 编码器：逐步下采样并保存中间特征
        for block in self.downs:
            x = block(x, t)
            residual_connections.append(x)

        # 解码器：取出对应层特征做跳跃连接，再逐步恢复分辨率
        for block in self.ups:
            residual_x = residual_connections.pop()

            # 尺寸可能不完全一致，先插值到当前特征图大小
            h, w = x.shape[2], x.shape[3]
            residual_x = F.interpolate(residual_x, size=(h, w), mode="nearest")
            x = torch.cat((x, residual_x), dim=1)
            x = block(x, t)

        # 输出前恢复到目标图像大小
        x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False)
        return self.final_conv(x)


# ===================== 3. DDIM 采样器 =====================
class DDIMSampler:
    def __init__(self, model, T, alphas_cumprod, device):
        self.model = model
        self.T = T
        self.alphas_cumprod = alphas_cumprod
        self.device = device

    def sample(self, image_size, batch_size=16, sample_steps=50, eta=0.0):
        """
        DDIM 跳步采样核心函数。

        :param image_size: 生成图像尺寸
        :param batch_size: 生成批大小
        :param sample_steps: 采样步数，通常远小于训练时的 T
        :param eta: 随机性系数，0 表示确定性 DDIM，越大越接近 DDPM
        :return: 采样后的图像张量
        """
        # 1. 从 [0, T-1] 中均匀取 sample_steps 个时间步，并按逆序采样
        step_sequence = (
            torch.linspace(0, self.T - 1, sample_steps).long().flip(dims=[0]).to(self.device)
        )
        total_steps = len(step_sequence)

        # 2. 从纯高斯噪声开始反向生成
        x = torch.randn((batch_size, img_channels, image_size, image_size), device=self.device)

        # 3. 依据 DDIM 更新公式逐步去噪
        for i in tqdm(range(total_steps - 1), desc=f"DDIM 采样中（{sample_steps}步）"):
            t = step_sequence[i]
            t_prev = step_sequence[i + 1]

            # 当前时间步与上一个采样时间步对应的 alpha_bar
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev]

            # 预测当前噪声 eps_theta(x_t, t)
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor)

            # 根据当前样本和预测噪声，反推原始图像估计 x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

            # eta 控制随机项强度；eta=0 时为确定性采样
            sigma_t = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )

            # 指向 x_{t-1} 的确定性方向项
            pred_dir = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * pred_noise

            # 最后一步不再额外加噪声
            if i < total_steps - 2:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # 得到上一个时间步的样本 x_{t-1}
            x = torch.sqrt(alpha_t_prev) * x0_pred + pred_dir + sigma_t * noise

        return x


# ===================== 4. 模型加载与采样 =====================
# 初始化模型结构，需与训练时完全一致
model = UNet().to(device)

# 这里加载训练好的 DDPM 权重
# model.load_state_dict(torch.load("ddpm_mnist.pth", map_location=device))
model.eval()

# 初始化 DDIM 采样器
ddim_sampler = DDIMSampler(model, T, alphas_cumprod, device)

# 执行 DDIM 采样
with torch.no_grad():
    # 50 步采样时 eta=0，为确定性生成，速度通常远快于 1000 步 DDPM
    samples = ddim_sampler.sample(img_size, batch_size=16, sample_steps=50, eta=0.0)

# 保存生成结果
samples = (samples + 1) / 2  # 将范围从 [-1, 1] 映射回 [0, 1]
np.save("ddim_samples.npy", samples.cpu().numpy())
print("DDIM 采样完成，样本已保存为 ddim_samples.npy")

# 可视化保存成 PNG，方便直接查看生成效果
try:
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(samples[i * 4 + j].squeeze().cpu().numpy(), cmap="gray")
            axs[i, j].axis("off")
    plt.savefig("ddim_samples_50steps.png", bbox_inches="tight")
    plt.close()
    print("PNG 图像已保存为 ddim_samples_50steps.png")
except Exception as e:
    print(f"保存 PNG 图像失败: {e}")
