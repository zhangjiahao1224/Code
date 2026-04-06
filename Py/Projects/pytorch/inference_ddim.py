import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ===================== 1. 关键参数 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
T = 1000
img_size = 28
img_channels = 1
script_dir = Path(__file__).resolve().parent
checkpoint_candidates = ["ddpm_unet_mnist.pth", "ddpm_mnist.pth"]


def resolve_checkpoint_path():
    """按候选文件名寻找权重文件，优先脚本所在目录。"""
    search_paths = [script_dir / name for name in checkpoint_candidates]

    # 兼容从其它工作目录启动脚本的场景
    if Path.cwd().resolve() != script_dir:
        search_paths.extend([Path.cwd() / name for name in checkpoint_candidates])

    for path in search_paths:
        if path.is_file():
            return path

    existing = sorted(p.name for p in script_dir.glob("*.pth"))
    candidate_text = ", ".join(checkpoint_candidates)
    existing_text = ", ".join(existing) if existing else "无"
    raise FileNotFoundError(
        f"未找到权重文件。请将以下任一文件放到目录 {script_dir}：{candidate_text}。"
        f"当前目录已有 .pth 文件：{existing_text}"
    )


# ===================== 2. 复用训练时的模型结构 =====================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # 将时间步编码为正余弦向量，让模型感知当前去噪阶段
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
            # 上采样阶段会和跳跃连接特征拼接，因此输入通道数翻倍
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
        # 卷积提取特征后，把时间嵌入加到特征图上
        h = self.relu(self.norm1(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(time_emb))
        h = h + time_emb[(...,) + (None,) * 2]
        h = self.relu(self.norm2(self.conv2(h)))
        return self.transform(h)


class UNet(nn.Module):
    def __init__(self, image_channels=1, time_emb_dim=128):
        super().__init__()
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
        residuals = []

        # 编码器：逐层下采样并保存中间特征
        for block in self.downs:
            x = block(x, t)
            residuals.append(x)

        # 解码器：拼接跳跃连接特征并逐步恢复分辨率
        for block in self.ups:
            residual_x = residuals.pop()
            h, w = x.shape[2], x.shape[3]
            residual_x = F.interpolate(residual_x, size=(h, w), mode="nearest")
            x = torch.cat((x, residual_x), dim=1)
            x = block(x, t)

        # 最终恢复到目标图像大小并输出预测噪声
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        return self.final_conv(x)


class DDIMSampler:
    def __init__(self, model, T, alphas_cumprod, device):
        self.model = model
        self.T = T
        self.alphas_cumprod = alphas_cumprod
        self.device = device

    def sample(self, image_size, batch_size=16, sample_steps=50, eta=0.0):
        # 在总扩散步中均匀选取若干时间步，按逆序执行 DDIM 采样
        step_sequence = (
            torch.linspace(0, self.T - 1, sample_steps).long().flip(dims=[0]).to(self.device)
        )
        x = torch.randn((batch_size, img_channels, image_size, image_size), device=self.device)

        # 从纯噪声开始，逐步迭代到更清晰的图像
        for i in tqdm(range(len(step_sequence) - 1), desc=f"DDIM 采样中（{sample_steps}步）"):
            t = step_sequence[i]
            t_prev = step_sequence[i + 1]
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev]

            # 利用网络预测当前噪声
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor)

            # 根据当前样本和预测噪声反推出原图估计 x_0
            x0_pred = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

            # eta 控制随机性，eta=0 时为确定性 DDIM
            sigma_t = eta * torch.sqrt(
                (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
            )

            # 确定性方向项，表示从 x_t 指向 x_{t-1} 的主要变化方向
            pred_dir = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * pred_noise

            # 最后一步不再补随机噪声
            noise = torch.randn_like(x) if i < len(step_sequence) - 2 else torch.zeros_like(x)

            # DDIM 更新公式，得到上一个时间步的样本
            x = torch.sqrt(alpha_t_prev) * x0_pred + pred_dir + sigma_t * noise
        return x


# ===================== 3. 推理主流程 =====================
# 构造与训练一致的噪声调度参数
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# 加载训练好的 UNet 权重
model = UNet().to(device)
checkpoint_path = resolve_checkpoint_path()
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
print(f"已加载权重: {checkpoint_path}")
model.eval()

# 初始化 DDIM 采样器
ddim = DDIMSampler(model, T, alphas_cumprod, device)

# 关闭梯度，执行采样推理
with torch.no_grad():
    samples = ddim.sample(img_size, batch_size=16, sample_steps=50, eta=0.0)

# 将输出从 [-1, 1] 映射回 [0, 1]，便于保存和显示
samples = (samples + 1) / 2
np.save("ddim_samples.npy", samples.cpu().numpy())
print("已保存 ddim_samples.npy")

# 保存前 16 张生成结果，便于快速查看采样效果
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(samples[i * 4 + j].squeeze().cpu().numpy(), cmap="gray")
        axs[i, j].axis("off")
plt.savefig("ddim_samples_50steps.png", bbox_inches="tight")
plt.close()
print("已保存 ddim_samples_50steps.png")
