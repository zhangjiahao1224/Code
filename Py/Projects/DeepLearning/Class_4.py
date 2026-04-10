import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


# ------------------------------------------------------------
# 本文件演示自编码器（Autoencoder）如何进行表示学习：
# 1. 先构造一个简单的灰度图形数据集；
# 2. 使用编码器把图像压缩到低维潜在向量 z；
# 3. 使用解码器根据 z 重建原图；
# 4. 用重建损失衡量“还原得像不像”；
# 5. 再额外用线性探针检查 latent representation 是否有分类价值。
# ------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """固定随机种子，让每次实验结果尽量可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """统计模型中可训练参数的总数量。"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _paint_square(img: torch.Tensor, x0: int, y0: int, size: int, intensity: float) -> None:
    """在图像上画一个实心正方形。"""

    img[y0 : y0 + size, x0 : x0 + size] = intensity


def _paint_hbar(img: torch.Tensor, x0: int, y0: int, width: int, thickness: int, intensity: float) -> None:
    """在图像上画一个横条形状。"""

    img[y0 : y0 + thickness, x0 : x0 + width] = intensity


def _paint_vbar(img: torch.Tensor, x0: int, y0: int, height: int, thickness: int, intensity: float) -> None:
    """在图像上画一个竖条形状。"""

    img[y0 : y0 + height, x0 : x0 + thickness] = intensity


def _paint_xshape(img: torch.Tensor, x0: int, y0: int, size: int, intensity: float) -> None:
    """在图像上画一个 X 形图案。"""

    for i in range(size):
        img[y0 + i, x0 + i] = intensity
        img[y0 + i, x0 + size - 1 - i] = intensity


def make_autoencoder_dataset(
    num_samples: int = 3600,
    image_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构造一个简单的灰度图数据集。

    这里虽然返回了 labels，但它们不是给自编码器训练用的，
    只是后面做“线性探针”时用来检查潜在表示 z 是否有区分能力。
    """

    # images 的形状为 [N, 1, H, W]，适合直接喂给卷积网络。
    images = torch.zeros(num_samples, 1, image_size, image_size, dtype=torch.float32)
    labels = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        # 随机选择 4 类图形中的一种：
        # 0 = square, 1 = horizontal bar, 2 = vertical bar, 3 = X-shape
        cls = random.randint(0, 3)
        labels[i] = cls

        # 先生成弱噪声背景，让数据不要过于理想化。
        img = 0.03 * torch.randn(image_size, image_size)
        img = img.clamp(0.0, 1.0)
        intensity = random.uniform(0.75, 1.0)

        if cls == 0:
            size = random.randint(5, 8)
            x0 = random.randint(1, image_size - size - 1)
            y0 = random.randint(1, image_size - size - 1)
            _paint_square(img, x0, y0, size, intensity)
        elif cls == 1:
            width = random.randint(8, 12)
            thickness = random.randint(2, 3)
            x0 = random.randint(1, image_size - width - 1)
            y0 = random.randint(1, image_size - thickness - 1)
            _paint_hbar(img, x0, y0, width, thickness, intensity)
        elif cls == 2:
            height = random.randint(8, 12)
            thickness = random.randint(2, 3)
            x0 = random.randint(1, image_size - thickness - 1)
            y0 = random.randint(1, image_size - height - 1)
            _paint_vbar(img, x0, y0, height, thickness, intensity)
        else:
            size = random.randint(7, 10)
            x0 = random.randint(1, image_size - size - 1)
            y0 = random.randint(1, image_size - size - 1)
            _paint_xshape(img, x0, y0, size, intensity)

        # 再做一次裁剪，确保所有像素都在 [0, 1] 内。
        images[i, 0] = img.clamp(0.0, 1.0)

    return images, labels


class ConvAutoencoder(nn.Module):
    """
    一个小型卷积自编码器。

    编码器负责压缩输入，解码器负责重建输入。
    其中 latent_dim 对应瓶颈层大小，越小代表压缩越强。
    """

    def __init__(self, image_size: int = 16, latent_dim: int = 16):
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size 必须能被 8 整除")

        self.image_size = image_size
        self.latent_dim = latent_dim
        # 三次 2x2 池化后，空间尺寸会缩小为原来的 1/8。
        self.feature_size = image_size // 8

        # 编码器：逐步提取图像特征，再压缩成低维潜在向量 z。
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64 * self.feature_size * self.feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        # 解码器：把低维 z 重新展开成特征图，再一步步上采样恢复成图像。
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64 * self.feature_size * self.feature_size),
            nn.ReLU(),
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """把输入图像编码为潜在表示 z。"""

        h = self.encoder_cnn(x)
        h = h.flatten(start_dim=1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """根据潜在表示 z 重建图像。"""

        h = self.decoder_fc(z)
        h = h.view(-1, 64, self.feature_size, self.feature_size)
        return self.decoder_cnn(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播同时返回：
        1. recon：重建结果
        2. z：潜在表示
        """

        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def evaluate_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """在验证集上计算逐像素平均重建误差（per-pixel MSE）。"""

    model.eval()
    total_squared_error = 0.0
    total_pixels = 0

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            # 这里统计的是“所有像素的平方误差总和”，
            # 最后再除以像素总数，得到更直观的逐像素平均 MSE。
            loss = F.mse_loss(recon, xb, reduction="sum")
            total_squared_error += loss.item()
            total_pixels += xb.numel()

    return total_squared_error / total_pixels


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """训练自编码器，并返回最佳验证集逐像素平均 MSE。"""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_squared_error = 0.0
        total_pixels = 0

        for xb, _ in train_loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            # v2 使用纯 MSE 重建损失，整体训练更稳定。
            loss = F.mse_loss(recon, xb)
            batch_squared_error = F.mse_loss(recon, xb, reduction="sum")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_squared_error += batch_squared_error.item()
            total_pixels += xb.numel()

        train_loss = running_squared_error / total_pixels
        val_loss = evaluate_autoencoder(model, val_loader, device)
        best_val_loss = min(best_val_loss, val_loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[自编码器] 第 {epoch:02d} 轮 | "
                f"训练集逐像素 MSE={train_loss:.6f} | 验证集逐像素 MSE={val_loss:.6f}"
            )

    return best_val_loss


@torch.no_grad()
def extract_latents(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把整批数据编码成潜在向量，并收集对应标签。"""

    model.eval()
    all_z = []
    all_y = []

    for xb, yb in loader:
        xb = xb.to(device)
        z = model.encode(xb).cpu()
        all_z.append(z)
        all_y.append(yb.cpu())

    return torch.cat(all_z, dim=0), torch.cat(all_y, dim=0)


def train_linear_probe(
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    val_z: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 5e-2,
) -> float:
    """
    在冻结的潜在表示 z 上训练一个线性分类器。

    这一步不是自编码器本身的训练目标，而是一个常见的分析方法：
    如果 z 很容易被线性层分开，说明它学到的表示更有结构、更有语义。
    """

    probe = nn.Linear(train_z.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_z = train_z.to(device)
    train_y = train_y.to(device)
    val_z = val_z.to(device)
    val_y = val_y.to(device)

    for epoch in range(1, epochs + 1):
        probe.train()
        logits = probe(train_z)
        loss = F.cross_entropy(logits, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            with torch.no_grad():
                val_acc = (probe(val_z).argmax(dim=1) == val_y).float().mean().item()
            print(f"[线性探针] 第 {epoch:03d} 轮 | 验证集准确率={val_acc:.4f}")

    probe.eval()
    with torch.no_grad():
        return (probe(val_z).argmax(dim=1) == val_y).float().mean().item()


def save_autoencoder_checkpoint(
    model: nn.Module,
    path: str = "artifacts/autoencoder_shapes.pt",
) -> str:
    """保存自编码器权重和必要的结构参数。"""

    ckpt = {
        "state_dict": model.state_dict(),
        "image_size": model.image_size,
        "latent_dim": model.latent_dim,
    }
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, file)
    return str(file)


def load_autoencoder_checkpoint(
    path: str,
    device: torch.device,
) -> ConvAutoencoder:
    """从磁盘加载之前保存好的自编码器。"""

    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ConvAutoencoder(
        image_size=int(ckpt["image_size"]),
        latent_dim=int(ckpt["latent_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def save_reconstruction_grid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    path: str = "artifacts/autoencoder_reconstructions.png",
    max_items: int = 8,
) -> str | None:
    """把“原图/重建图”拼成网格图，方便直观看训练效果。"""

    try:
        from PIL import Image
    except ImportError:
        print("未安装 Pillow，跳过重建结果图片导出。")
        return None

    # 取一个 batch 的前 max_items 张图进行可视化。
    xb, _ = next(iter(loader))
    xb = xb[:max_items].to(device)
    recon, _ = model(xb)

    originals = xb.cpu()
    reconstructions = recon.cpu()
    image_size = originals.size(-1)
    scale = 16
    # 上半部分放原图，下半部分放对应的重建图。
    canvas = Image.new("L", (max_items * image_size * scale, 2 * image_size * scale), color=255)

    for idx in range(max_items):
        top = originals[idx, 0].mul(255).clamp(0, 255).byte().numpy()
        bottom = reconstructions[idx, 0].mul(255).clamp(0, 255).byte().numpy()

        top_img = Image.fromarray(top, mode="L").resize(
            (image_size * scale, image_size * scale),
            Image.Resampling.NEAREST,
        )
        bottom_img = Image.fromarray(bottom, mode="L").resize(
            (image_size * scale, image_size * scale),
            Image.Resampling.NEAREST,
        )

        canvas.paste(top_img, (idx * image_size * scale, 0))
        canvas.paste(bottom_img, (idx * image_size * scale, image_size * scale))

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(file)
    return str(file)


@torch.no_grad()
def save_tensor_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    scale: int = 16,
) -> str | None:
    """
    把一批单通道图像按网格保存。

    这个工具会同时给 VAE 采样结果、latent traversal 和 GAN 生成样本复用。
    """

    try:
        from PIL import Image
    except ImportError:
        print("未安装 Pillow，跳过图像网格导出。")
        return None

    images = images.detach().cpu().clamp(0.0, 1.0)
    num_images, _, image_size, _ = images.shape
    nrow = max(1, min(nrow, num_images))
    ncol = (num_images + nrow - 1) // nrow
    canvas = Image.new(
        "L",
        (nrow * image_size * scale, ncol * image_size * scale),
        color=255,
    )

    for idx in range(num_images):
        row = idx // nrow
        col = idx % nrow
        arr = images[idx, 0].mul(255).byte().numpy()
        tile = Image.fromarray(arr, mode="L").resize(
            (image_size * scale, image_size * scale),
            Image.Resampling.NEAREST,
        )
        canvas.paste(tile, (col * image_size * scale, row * image_size * scale))

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(file)
    return str(file)


class ConvVAE(nn.Module):
    """
    最简卷积 VAE。

    与普通 Autoencoder 的区别在于：
    1. 编码器不直接输出一个固定的 z；
    2. 而是输出分布参数 mu 和 logvar；
    3. 再通过重参数化技巧采样得到 z。
    """

    def __init__(self, image_size: int = 16, latent_dim: int = 8):
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size 必须能被 8 整除")

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_size = image_size // 8

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(64 * self.feature_size * self.feature_size, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64 * self.feature_size * self.feature_size),
            nn.ReLU(),
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """编码器输出均值 mu 和对数方差 logvar。"""

        h = self.encoder_cnn(x)
        h = h.flatten(start_dim=1)
        h = self.encoder_fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：
        z = mu + sigma * epsilon，其中 epsilon ~ N(0, 1)

        这样随机性来自 epsilon，而 mu / logvar 仍然保持可导。
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """根据潜在变量 z 重建图像。"""

        h = self.decoder_fc(z)
        h = h.view(-1, 64, self.feature_size, self.feature_size)
        return self.decoder_cnn(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回重建结果、采样后的 z、mu、logvar。"""

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE 总损失 = 重建项 + beta * KL 项。

    - 重建项：让输出尽量像输入
    - KL 项：让后验 q(z|x) 接近标准正态先验 p(z)=N(0, I)
    """

    recon_sum = F.mse_loss(recon, x, reduction="sum")
    kl_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    batch_size = x.size(0)
    loss = (recon_sum + beta * kl_sum) / batch_size
    return loss, recon_sum / x.numel(), kl_sum / batch_size


def evaluate_vae(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    beta: float = 1.0,
) -> tuple[float, float, float]:
    """返回 VAE 的平均总损失、逐像素 MSE 与每样本 KL。"""

    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_batches = 0

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            recon, _, mu, logvar = model(xb)
            loss, recon_mse, kl_per_sample = vae_loss(recon, xb, mu, logvar, beta=beta)
            total_loss += loss.item()
            total_recon += recon_mse.item()
            total_kl += kl_per_sample.item()
            total_batches += 1

    return (
        total_loss / total_batches,
        total_recon / total_batches,
        total_kl / total_batches,
    )


def train_vae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    beta: float = 1.0,
) -> tuple[float, float]:
    """训练 VAE，并返回最佳验证集总损失与逐像素 MSE。"""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_val_recon = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_batches = 0

        for xb, _ in train_loader:
            xb = xb.to(device)
            recon, _, mu, logvar = model(xb)
            loss, recon_mse, kl_per_sample = vae_loss(recon, xb, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_mse.item()
            total_kl += kl_per_sample.item()
            total_batches += 1

        train_loss = total_loss / total_batches
        train_recon = total_recon / total_batches
        train_kl = total_kl / total_batches
        val_loss, val_recon, val_kl = evaluate_vae(model, val_loader, device, beta=beta)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_recon = val_recon

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[VAE] 第 {epoch:02d} 轮 | "
                f"训练总损失={train_loss:.4f} | 训练重建MSE={train_recon:.6f} | "
                f"训练KL={train_kl:.4f} | 验证重建MSE={val_recon:.6f} | 验证KL={val_kl:.4f}"
            )

    return best_val_loss, best_val_recon


def save_vae_checkpoint(
    model: nn.Module,
    path: str = "artifacts/vae_shapes.pt",
) -> str:
    """保存 VAE 权重。"""

    ckpt = {
        "state_dict": model.state_dict(),
        "image_size": model.image_size,
        "latent_dim": model.latent_dim,
    }
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, file)
    return str(file)


def load_vae_checkpoint(path: str, device: torch.device) -> ConvVAE:
    """加载 VAE 权重。"""

    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ConvVAE(
        image_size=int(ckpt["image_size"]),
        latent_dim=int(ckpt["latent_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def save_vae_reconstruction_grid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    path: str = "artifacts/vae_reconstructions.png",
    max_items: int = 8,
) -> str | None:
    """保存 VAE 的原图/重建图对比。"""

    try:
        from PIL import Image
    except ImportError:
        print("未安装 Pillow，跳过 VAE 重建图导出。")
        return None

    xb, _ = next(iter(loader))
    xb = xb[:max_items].to(device)
    recon, _, _, _ = model(xb)
    originals = xb.cpu()
    reconstructions = recon.cpu()
    image_size = originals.size(-1)
    scale = 16
    canvas = Image.new("L", (max_items * image_size * scale, 2 * image_size * scale), color=255)

    for idx in range(max_items):
        top = originals[idx, 0].mul(255).byte().numpy()
        bottom = reconstructions[idx, 0].mul(255).byte().numpy()
        top_img = Image.fromarray(top, mode="L").resize(
            (image_size * scale, image_size * scale),
            Image.Resampling.NEAREST,
        )
        bottom_img = Image.fromarray(bottom, mode="L").resize(
            (image_size * scale, image_size * scale),
            Image.Resampling.NEAREST,
        )
        canvas.paste(top_img, (idx * image_size * scale, 0))
        canvas.paste(bottom_img, (idx * image_size * scale, image_size * scale))

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(file)
    return str(file)


@torch.no_grad()
def save_vae_generated_samples(
    model: nn.Module,
    device: torch.device,
    path: str = "artifacts/vae_samples.png",
    num_samples: int = 16,
) -> str | None:
    """从标准正态分布采样 z，再由 VAE 解码生成新样本。"""

    z = torch.randn(num_samples, model.latent_dim, device=device)
    samples = model.decode(z)
    return save_tensor_image_grid(samples, path=path, nrow=4)


@torch.no_grad()
def save_vae_latent_traversal(
    model: nn.Module,
    device: torch.device,
    path: str = "artifacts/vae_latent_traversal.png",
    num_steps: int = 7,
) -> str | None:
    """
    固定其他 latent 维度，仅改变前两个维度，观察生成结果如何变化。

    这能帮助理解 latent perturbation 和 disentanglement 的直观含义。
    """

    values = torch.linspace(-2.0, 2.0, steps=num_steps, device=device)
    samples = []
    for v1 in values:
        for v2 in values:
            z = torch.zeros(1, model.latent_dim, device=device)
            if model.latent_dim >= 1:
                z[0, 0] = v1
            if model.latent_dim >= 2:
                z[0, 1] = v2
            samples.append(model.decode(z).cpu())

    grid = torch.cat(samples, dim=0)
    return save_tensor_image_grid(grid, path=path, nrow=num_steps)


class MLPGenerator(nn.Module):
    """
    最简 GAN 生成器。

    输入随机噪声 z，输出一张 16x16 的单通道图像。
    """

    def __init__(self, noise_dim: int = 32, image_size: int = 16):
        super().__init__()
        self.noise_dim = noise_dim
        self.image_size = image_size
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, image_size * image_size),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        return x.view(-1, 1, self.image_size, self.image_size)


class MLPDiscriminator(nn.Module):
    """
    最简 GAN 判别器。

    输入一张图像，输出一个 logit，表示“像真样本”的程度。
    """

    def __init__(self, image_size: int = 16):
        super().__init__()
        self.image_size = image_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    noise_dim: int = 32,
) -> tuple[float, float]:
    """
    训练最简 GAN。

    - 判别器 D：尽量把真实样本判为真，把生成样本判为假
    - 生成器 G：尽量骗过判别器，让假样本也被判成真
    """

    generator.to(device)
    discriminator.to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    last_d_loss = 0.0
    last_g_loss = 0.0

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        running_d = 0.0
        running_g = 0.0
        total_batches = 0

        for (xb,) in loader:
            xb = xb.to(device)
            bs = xb.size(0)

            real_targets = torch.ones(bs, 1, device=device)
            fake_targets = torch.zeros(bs, 1, device=device)

            # 先训练判别器：提高“分真假”的能力。
            z = torch.randn(bs, noise_dim, device=device)
            fake_images = generator(z).detach()
            real_logits = discriminator(xb)
            fake_logits = discriminator(fake_images)
            d_loss_real = loss_fn(real_logits, real_targets)
            d_loss_fake = loss_fn(fake_logits, fake_targets)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # 再训练生成器：让判别器把假图也当成真图。
            z = torch.randn(bs, noise_dim, device=device)
            fake_images = generator(z)
            fake_logits = discriminator(fake_images)
            g_loss = loss_fn(fake_logits, real_targets)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            running_d += d_loss.item()
            running_g += g_loss.item()
            total_batches += 1

        last_d_loss = running_d / total_batches
        last_g_loss = running_g / total_batches
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[GAN] 第 {epoch:03d} 轮 | "
                f"判别器损失={last_d_loss:.4f} | 生成器损失={last_g_loss:.4f}"
            )

    return last_d_loss, last_g_loss


def save_gan_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    path: str = "artifacts/gan_shapes.pt",
) -> str:
    """保存 GAN 的生成器和判别器权重。"""

    ckpt = {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "noise_dim": generator.noise_dim,
        "image_size": generator.image_size,
    }
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, file)
    return str(file)


def load_gan_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[MLPGenerator, MLPDiscriminator]:
    """加载 GAN 权重。"""

    ckpt = torch.load(path, map_location=device, weights_only=True)
    generator = MLPGenerator(
        noise_dim=int(ckpt["noise_dim"]),
        image_size=int(ckpt["image_size"]),
    ).to(device)
    discriminator = MLPDiscriminator(image_size=int(ckpt["image_size"])).to(device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    discriminator.load_state_dict(ckpt["discriminator_state_dict"])
    generator.eval()
    discriminator.eval()
    return generator, discriminator


@torch.no_grad()
def save_gan_generated_samples(
    generator: nn.Module,
    device: torch.device,
    path: str = "artifacts/gan_samples.png",
    num_samples: int = 16,
) -> str | None:
    """从随机噪声采样，并导出 GAN 生成样本。"""

    z = torch.randn(num_samples, generator.noise_dim, device=device)
    samples = generator(z)
    return save_tensor_image_grid(samples, path=path, nrow=4)

def compute_pca_2d(latents: torch.Tensor) -> torch.Tensor:
    """
    使用 PCA 把高维 latent 向量降到二维。

    这里直接用 torch 做线性代数分解，避免依赖额外库。
    """

    x = latents.float()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=2)
    return x @ v[:, :2]


def compute_tsne_2d(latents: torch.Tensor) -> torch.Tensor | None:
    """
    尝试使用 t-SNE 做二维降维。

    如果环境中没有安装 scikit-learn，就返回 None，
    这样整份脚本仍然可以继续运行。
    """

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return None

    latents_np = latents.float().cpu().numpy()
    perplexity = min(30, max(5, latents_np.shape[0] // 100))
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=42,
    )
    points = tsne.fit_transform(latents_np)
    return torch.from_numpy(points).float()


def save_latent_scatter_plot(
    points_2d: torch.Tensor,
    labels: torch.Tensor,
    class_names: tuple[str, ...],
    path: str,
    title: str,
) -> str | None:
    """把二维 latent 分布画成散点图并保存。"""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("未安装 Pillow，跳过 latent 可视化图片导出。")
        return None

    # 颜色分别对应 4 类图形，方便观察类间是否自然分开。
    palette = [
        (230, 57, 70),
        (29, 78, 216),
        (46, 125, 50),
        (245, 158, 11),
    ]

    width, height = 900, 700
    margin = 70
    legend_x = width - 180
    plot_left = margin
    plot_top = margin
    plot_right = legend_x - 20
    plot_bottom = height - margin

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    x_vals = points_2d[:, 0]
    y_vals = points_2d[:, 1]
    x_min = float(x_vals.min().item())
    x_max = float(x_vals.max().item())
    y_min = float(y_vals.min().item())
    y_max = float(y_vals.max().item())

    # 避免所有点过于集中时分母为 0。
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    def project(px: float, py: float) -> tuple[int, int]:
        x = plot_left + int((px - x_min) / x_span * (plot_right - plot_left))
        y = plot_bottom - int((py - y_min) / y_span * (plot_bottom - plot_top))
        return x, y

    draw.text((margin, 20), title, fill="black", font=font)
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black", width=2)
    draw.text((plot_left, plot_bottom + 10), "dim-1", fill="gray", font=font)
    draw.text((plot_left - 40, plot_top - 20), "dim-2", fill="gray", font=font)

    for point, label in zip(points_2d, labels):
        x, y = project(float(point[0].item()), float(point[1].item()))
        color = palette[int(label.item()) % len(palette)]
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color, outline=color)

    draw.text((legend_x, plot_top), "Legend", fill="black", font=font)
    for idx, name in enumerate(class_names):
        y = plot_top + 30 + idx * 28
        color = palette[idx % len(palette)]
        draw.rectangle([legend_x, y, legend_x + 14, y + 14], fill=color, outline=color)
        draw.text((legend_x + 24, y), name, fill="black", font=font)

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    img.save(file)
    return str(file)


def save_latent_visualizations(
    latents: torch.Tensor,
    labels: torch.Tensor,
    class_names: tuple[str, ...],
    max_points: int = 1200,
) -> tuple[str | None, str | None]:
    """
    保存 latent 空间的 PCA / t-SNE 可视化。

    为了让图更清晰，默认只抽样一部分点来画。
    """

    if latents.size(0) > max_points:
        indices = torch.randperm(latents.size(0))[:max_points]
        latents = latents[indices]
        labels = labels[indices]

    pca_points = compute_pca_2d(latents)
    pca_path = save_latent_scatter_plot(
        pca_points,
        labels,
        class_names,
        path="artifacts/latent_pca.png",
        title="Latent Space Visualization (PCA)",
    )

    tsne_points = compute_tsne_2d(latents)
    tsne_path = None
    if tsne_points is not None:
        tsne_path = save_latent_scatter_plot(
            tsne_points,
            labels,
            class_names,
            path="artifacts/latent_tsne.png",
            title="Latent Space Visualization (t-SNE)",
        )

    return pca_path, tsne_path


def run_autoencoder_lesson(device: torch.device, force_retrain: bool = False) -> None:
    """串联完整流程：准备数据、训练模型、评估表示并保存结果。"""

    image_size = 16
    latent_dim = 16
    batch_size = 128
    ckpt_path = "artifacts/autoencoder_shapes_v2.pt"
    class_names = ("square", "h_bar", "v_bar", "x_shape")

    images, labels = make_autoencoder_dataset(num_samples=3600, image_size=image_size)
    dataset = TensorDataset(images, labels)

    # 按 8:2 划分训练集和验证集。
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    full_loader = DataLoader(dataset, batch_size=batch_size)

    # 输入维度 / latent_dim 越大，说明压缩得越厉害。
    compression_ratio = (image_size * image_size) / latent_dim
    print("自编码器表示学习实验")
    print(f"- 输入维度：{image_size * image_size}")
    print(f"- 潜在向量维度：{latent_dim}")
    print(f"- 压缩倍率：{compression_ratio:.1f}x")

    if Path(ckpt_path).exists() and not force_retrain:
        print(f"检测到已有权重，直接加载：{ckpt_path}")
        model = load_autoencoder_checkpoint(ckpt_path, device)
        best_val_loss = evaluate_autoencoder(model, val_loader, device)
    else:
        model = ConvAutoencoder(image_size=image_size, latent_dim=latent_dim)
        print(f"开始训练新的自编码器，参数量：{count_parameters(model)}")
        best_val_loss = train_autoencoder(
            model,
            train_loader,
            val_loader,
            epochs=35,
            lr=1e-3,
            device=device,
        )
        save_autoencoder_checkpoint(model, ckpt_path)

    # 保存可视化重建结果，看看模型是不是学会了保留主要结构。
    recon_path = save_reconstruction_grid(model, val_loader, device)
    train_latents, train_targets = extract_latents(model, train_loader, device)
    val_latents, val_targets = extract_latents(model, val_loader, device)
    full_latents, full_targets = extract_latents(model, full_loader, device)
    probe_acc = train_linear_probe(
        train_latents,
        train_targets,
        val_latents,
        val_targets,
        num_classes=4,
        device=device,
    )
    pca_path, tsne_path = save_latent_visualizations(full_latents, full_targets, class_names)

    # -----------------------------
    # 实验结果分析：
    # 1. 如果逐像素 MSE 持续下降，说明解码器越来越能根据 z 还原输入；
    # 2. 如果线性探针准确率较高，说明 z 中已经包含较清晰的类别信息；
    # 3. 如果 PCA / t-SNE 图中不同颜色逐渐分开，说明 latent space
    #    在无监督重建目标下，自发形成了更有结构的聚类分布。
    # 4. 当前回到 v2 版本，使用纯 MSE 作为训练目标，
    #    更适合这个简化图形数据集的稳定重建。
    # -----------------------------
    print(f"最佳验证集逐像素 MSE：{best_val_loss:.6f}")
    print(f"冻结潜在向量后的线性探针准确率：{probe_acc:.4f}")
    if recon_path is not None:
        print(f"重建效果图片已保存到：{recon_path}")
    if pca_path is not None:
        print(f"PCA latent 可视化已保存到：{pca_path}")
    if tsne_path is not None:
        print(f"t-SNE latent 可视化已保存到：{tsne_path}")
    else:
        print("未生成 t-SNE 可视化：当前环境可能未安装 scikit-learn。")

    with torch.no_grad():
        full_recon_loss = evaluate_autoencoder(model, full_loader, device)
    print(f"全数据集逐像素 MSE：{full_recon_loss:.6f}")
    print("结果解读：")
    print("- 瓶颈层会强迫模型把每张图压缩成较短的潜在向量。")
    print("- 重建损失会推动这个潜在向量尽量保留还原原图所需的信息。")
    print("- 线性探针用于检查这个潜在表示是否也对下游图形分类有帮助。")


def run_vae_lesson(device: torch.device, force_retrain: bool = False) -> None:
    """
    第四节课的第二部分：VAE。

    这里重点演示：
    1. 编码器输出的是分布参数 mu / logvar；
    2. 用重参数化技巧采样 z；
    3. 用 reconstruction + KL 共同训练；
    4. 观察 VAE 的采样和 latent perturbation 效果。
    """

    image_size = 16
    latent_dim = 8
    batch_size = 128
    beta = 1.0
    ckpt_path = "artifacts/vae_shapes.pt"

    images, labels = make_autoencoder_dataset(num_samples=3600, image_size=image_size)
    dataset = TensorDataset(images, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    print("\n" + "=" * 68)
    print("VAE 概率生成实验")
    print(f"- 图像尺寸：{image_size}x{image_size}")
    print(f"- 潜在向量维度：{latent_dim}")
    print(f"- beta 系数：{beta}")

    if Path(ckpt_path).exists() and not force_retrain:
        print(f"检测到已有 VAE 权重，直接加载：{ckpt_path}")
        model = load_vae_checkpoint(ckpt_path, device)
        best_val_loss, best_val_recon, best_val_kl = evaluate_vae(model, val_loader, device, beta=beta)
    else:
        model = ConvVAE(image_size=image_size, latent_dim=latent_dim)
        print(f"开始训练新的 VAE，参数量：{count_parameters(model)}")
        best_val_loss, best_val_recon = train_vae(
            model,
            train_loader,
            val_loader,
            epochs=25,
            lr=1e-3,
            device=device,
            beta=beta,
        )
        save_vae_checkpoint(model, ckpt_path)
        _, _, best_val_kl = evaluate_vae(model, val_loader, device, beta=beta)

    recon_path = save_vae_reconstruction_grid(model, val_loader, device)
    sample_path = save_vae_generated_samples(model, device)
    traversal_path = save_vae_latent_traversal(model, device)

    print(f"VAE 最佳验证集总损失：{best_val_loss:.4f}")
    print(f"VAE 验证集逐像素重建 MSE：{best_val_recon:.6f}")
    print(f"VAE 验证集每样本 KL：{best_val_kl:.4f}")
    if recon_path is not None:
        print(f"VAE 重建图已保存到：{recon_path}")
    if sample_path is not None:
        print(f"VAE 随机采样图已保存到：{sample_path}")
    if traversal_path is not None:
        print(f"VAE latent traversal 图已保存到：{traversal_path}")
    print("结果解读：")
    print("- 普通自编码器学习的是一个确定性的 z。")
    print("- VAE 学习的是 q(z|x) 的分布参数，因此 latent space 更适合采样。")
    print("- 重参数化技巧让采样层仍然可以参与反向传播。")
    print("- KL 项会把后验分布往标准正态先验拉近，从而让生成更平滑。")


def run_gan_lesson(device: torch.device, force_retrain: bool = False) -> None:
    """
    第四节课的第三部分：GAN。

    这里使用一个最简 MLP 版 GAN，重点理解：
    1. Generator 负责“造假”
    2. Discriminator 负责“打假”
    3. 两个网络通过对抗训练共同提升
    """

    image_size = 16
    noise_dim = 32
    batch_size = 128
    ckpt_path = "artifacts/gan_shapes.pt"

    images, _ = make_autoencoder_dataset(num_samples=3600, image_size=image_size)
    dataset = TensorDataset(images)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n" + "=" * 68)
    print("GAN 对抗生成实验")
    print(f"- 图像尺寸：{image_size}x{image_size}")
    print(f"- 噪声向量维度：{noise_dim}")

    if Path(ckpt_path).exists() and not force_retrain:
        print(f"检测到已有 GAN 权重，直接加载：{ckpt_path}")
        generator, discriminator = load_gan_checkpoint(ckpt_path, device)
        last_d_loss = float("nan")
        last_g_loss = float("nan")
    else:
        generator = MLPGenerator(noise_dim=noise_dim, image_size=image_size)
        discriminator = MLPDiscriminator(image_size=image_size)
        print(
            f"开始训练新的 GAN，生成器参数量：{count_parameters(generator)}，"
            f"判别器参数量：{count_parameters(discriminator)}"
        )
        last_d_loss, last_g_loss = train_gan(
            generator,
            discriminator,
            train_loader,
            epochs=40,
            lr=2e-4,
            device=device,
            noise_dim=noise_dim,
        )
        save_gan_checkpoint(generator, discriminator, ckpt_path)

    sample_path = save_gan_generated_samples(generator, device)
    if sample_path is not None:
        print(f"GAN 生成样本图已保存到：{sample_path}")
    if last_d_loss == last_d_loss and last_g_loss == last_g_loss:
        print(f"GAN 最后一轮判别器损失：{last_d_loss:.4f}")
        print(f"GAN 最后一轮生成器损失：{last_g_loss:.4f}")
    print("结果解读：")
    print("- Generator 从简单噪声 z 出发，学习生成像真的图像。")
    print("- Discriminator 接收真实图和生成图，学习区分真假。")
    print("- 当两个网络对抗训练时，生成器会逐渐学会更逼真的分布映射。")
    print("- GAN 不依赖显式重建损失，而是依赖“造假 vs 打假”的博弈。")


def main() -> None:
    """程序入口。"""

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备：{device}")
    run_autoencoder_lesson(device=device, force_retrain=False)
    run_vae_lesson(device=device, force_retrain=False)
    run_gan_lesson(device=device, force_retrain=False)


if __name__ == "__main__":
    main()
