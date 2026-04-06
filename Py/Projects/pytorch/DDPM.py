import os

# 避免 Windows 下 OpenMP 运行时重复加载导致程序中断
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== 1. 设备配置 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# ===================== 2. 超参数配置 =====================
T = 1000  # 总扩散步数
batch_size = 128
epochs = 50
lr = 1e-4
img_size = 28
img_channels = 1

# 噪声调度（线性调度）
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

# 预计算前向扩散需要的参数
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 预计算反向去噪需要的参数
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# ===================== 3. 时间步编码 =====================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# ===================== 4. 简化版U-Net去噪网络 =====================
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, time_emb):
        # 卷积+归一化+激活
        h = self.relu(self.norm1(self.conv1(x)))
        # 加入时间编码
        time_emb = self.relu(self.time_mlp(time_emb))
        h = h + time_emb[(..., ) + (None, ) * 2]
        # 第二层卷积
        h = self.relu(self.norm2(self.conv2(h)))
        # 下采样/上采样
        return self.transform(h)

class UNet(nn.Module):
    def __init__(self, image_channels=1, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        # 时间编码层
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        # 初始卷积
        self.init_conv = nn.Conv2d(image_channels, 64, 3, padding=1)
        # 下采样编码器
        self.downs = nn.ModuleList([
            Block(64, 64, time_emb_dim),
            Block(64, 128, time_emb_dim),
            Block(128, 256, time_emb_dim),
        ])
        # 上采样解码器
        self.ups = nn.ModuleList([
            Block(256, 128, time_emb_dim, up=True),
            Block(128, 64, time_emb_dim, up=True),
            Block(64, 64, time_emb_dim, up=True),
        ])
        # 最终输出卷积（输出和输入同尺寸的噪声图）
        self.final_conv = nn.Conv2d(64, image_channels, 1)

    def forward(self, x, timestep):
        # 时间编码
        t = self.time_mlp(timestep)
        # 初始卷积
        x = self.init_conv(x)
        # 保存跳跃连接的特征
        residual_connections = []
        # 下采样
        for block in self.downs:
            x = block(x, t)
            residual_connections.append(x)
        # 上采样+跳跃连接
        for block in self.ups:
            residual_x = residual_connections.pop()
            # 裁剪残差连接以匹配当前x的尺寸
            h, w = x.shape[2], x.shape[3]
            residual_x = residual_x[:, :, :h, :w]
            x = torch.cat((x, residual_x), dim=1)
            x = block(x, t)
        # 插值到原始尺寸
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        # 最终输出
        return self.final_conv(x)

# ===================== 5. DDPM核心类 =====================
class DDPM:
    def __init__(self, model, T, device):
        self.model = model
        self.T = T
        self.device = device

    # 前向扩散：生成x_t
    def q_sample(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise, noise

    # 单步反向去噪
    def p_sample(self, x, t):
        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        # 预测噪声
        pred_noise = self.model(x, t_tensor)
        # 从预测噪声计算预测的x_start
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        pred_x_start = (x - sqrt_one_minus_alpha_cumprod_t * pred_noise) / sqrt_alpha_cumprod_t
        # 计算反向过程的均值
        coef1 = posterior_mean_coef1[t].reshape(-1, 1, 1, 1)
        coef2 = posterior_mean_coef2[t].reshape(-1, 1, 1, 1)
        mu = coef1 * x + coef2 * pred_x_start
        # 计算方差
        sigma_t = torch.sqrt(posterior_variance[t])
        # 添加噪声
        if t == 0:
            return mu
        else:
            noise = torch.randn_like(x)
            return mu + sigma_t * noise

    # 完整采样生成
    def sample(self, image_size, batch_size=16):
        # 初始纯噪声
        x = torch.randn((batch_size, img_channels, image_size, image_size), device=self.device)
        # 反向迭代去噪
        for t in tqdm(reversed(range(0, self.T)), desc="采样中"):
            x = self.p_sample(x, t)
        return x

# ===================== 6. 数据加载 =====================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]，和高斯噪声分布匹配
])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===================== 7. 模型初始化 =====================
import math
model = UNet().to(device)
ddpm = DDPM(model, T, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# ===================== 8. 训练循环 =====================
print("开始训练...")
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch, _ in pbar:
        batch = batch.to(device)
        batch_size = batch.shape[0]
        
        # 随机采样时间步t
        t = torch.randint(0, T, (batch_size,), device=device).long()
        
        # 前向加噪
        x_noisy, noise = ddpm.q_sample(batch, t)
        
        # 预测噪声
        pred_noise = model(x_noisy, t)
        
        # 计算损失
        loss = criterion(pred_noise, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    # 每5个epoch采样一次
    if (epoch+1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            samples = ddpm.sample(img_size, batch_size=16)
        # 可视化生成结果
        samples = (samples + 1) / 2  # 从[-1,1]还原到[0,1]
        try:
            fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            for i in range(4):
                for j in range(4):
                    axs[i,j].imshow(samples[i*4+j].squeeze().cpu().numpy(), cmap="gray")
                    axs[i,j].axis("off")
            plt.savefig(f"ddpm_samples_epoch_{epoch+1}.png")
            plt.close()
        except Exception as e:
            print(f"保存图像失败: {e}")
        model.train()

# 训练结束后保存UNet权重供DDIM推理使用
torch.save(model.state_dict(), "ddpm_unet_mnist.pth")
print("已保存 ddpm_unet_mnist.pth")
print("训练完成！")
