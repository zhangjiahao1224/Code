import copy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


def make_synthetic_binary_dataset(num_samples=4000, input_dim=20):
    """生成一个简单的二分类合成数据集。"""
    # 随机生成特征。
    x = torch.randn(num_samples, input_dim)
    # 使用隐藏线性规则并叠加噪声，模拟真实数据分布。
    true_w = torch.randn(input_dim, 1)
    logits = x @ true_w + 0.25 * torch.randn(num_samples, 1)
    # 将概率转换为二值标签 {0, 1}。
    y = (torch.sigmoid(logits) > 0.5).float()
    return x, y


def augment_batch(x, noise_std=0.05, feature_drop_prob=0.05):
    """对表格类张量做轻量级数据增强。"""
    # 添加高斯噪声。
    noise = torch.randn_like(x) * noise_std
    # 随机丢弃部分特征（输入层正则化）。
    keep_mask = (torch.rand_like(x) > feature_drop_prob).float()
    return (x + noise) * keep_mask


class SmallMLP(nn.Module):
    """适用于小中型玩具任务的紧凑 MLP 模型。"""

    def __init__(self, input_dim, hidden1=32, hidden2=16, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model):
    # 仅统计可训练参数数量。
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_model_config(model):
    """从 SmallMLP 实例中提取结构配置。"""
    return {
        "input_dim": model.net[0].in_features,
        "hidden1": model.net[0].out_features,
        "hidden2": model.net[3].out_features,
        "dropout_p": model.net[2].p,
    }


def save_checkpoint(model, history, checkpoint_path="artifacts/best_mlp_checkpoint.pt"):
    """保存模型权重、结构配置和训练历史。"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": extract_model_config(model),
        "history": history,
    }
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_file)
    return str(checkpoint_file)


def load_checkpoint(checkpoint_path, device=None):
    """从检查点加载模型，并恢复对应结构。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["model_config"]

    model = SmallMLP(
        input_dim=config["input_dim"],
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        dropout_p=config["dropout_p"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("history")


def plot_history(history, save_path="artifacts/training_curves.png"):
    """使用 Pillow 绘制并保存训练/验证曲线。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow 未安装，跳过训练曲线绘制。")
        return None

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    if not train_loss or not val_loss or not val_acc:
        print("history 为空，跳过训练曲线绘制。")
        return None

    width, height = 1200, 520
    margin = 50
    title_h = 40
    panel_gap = 30
    panel_w = (width - margin * 2 - panel_gap) // 2
    panel_h = height - margin - title_h - 40
    panel_top = title_h + 25

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    def to_points(values, x0, y0, w, h, y_min, y_max):
        if len(values) == 1:
            return [(x0, y0 + h // 2)]
        denom = max(y_max - y_min, 1e-12)
        points = []
        for i, v in enumerate(values):
            x = x0 + int(i * w / (len(values) - 1))
            y = y0 + h - int((v - y_min) * h / denom)
            points.append((x, y))
        return points

    def draw_panel(x0, title, series, y_min, y_max):
        y0 = panel_top
        draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], outline="black", width=1)
        draw.text((x0, y0 - 18), title, fill="black", font=font)

        for idx, (name, values, color) in enumerate(series):
            pts = to_points(values, x0 + 8, y0 + 8, panel_w - 16, panel_h - 16, y_min, y_max)
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)
            draw.text((x0 + 8, y0 + panel_h + 8 + 12 * idx), name, fill=color, font=font)

        draw.text((x0 + panel_w - 120, y0 + panel_h + 8), f"min={y_min:.4f}", fill="gray", font=font)
        draw.text((x0 + panel_w - 120, y0 + panel_h + 20), f"max={y_max:.4f}", fill="gray", font=font)

    loss_all = train_loss + val_loss
    loss_min = min(loss_all)
    loss_max = max(loss_all)
    if abs(loss_max - loss_min) < 1e-12:
        loss_max = loss_min + 1e-6

    draw.text((margin, 10), "Training Curves", fill="black", font=font)
    draw_panel(
        margin,
        "Loss",
        [("Train Loss", train_loss, "blue"), ("Val Loss", val_loss, "red")],
        loss_min,
        loss_max,
    )
    draw_panel(
        margin + panel_w + panel_gap,
        "Validation Accuracy",
        [("Val Acc", val_acc, "green")],
        0.0,
        1.0,
    )

    plot_file = Path(save_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    img.save(plot_file)
    return str(plot_file)


def evaluate(model, loader, loss_fn, device):
    # 评估模式会关闭 Dropout 的随机失活行为。
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 验证/推理阶段关闭梯度跟踪以节省开销。
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total_loss += loss.item() * xb.size(0)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

    # 返回平均损失与准确率。
    return total_loss / total_samples, total_correct / total_samples


def train_with_regularization(
    num_samples=4000,
    input_dim=20,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=100,
    patience=8,
    min_delta=1e-4,
    val_ratio=0.2,
    noise_std=0.05,
    feature_drop_prob=0.05,
    seed=42,
):
    """包含数据增强 + AdamW + 早停机制的训练流程。"""
    # 固定随机种子，保证实验可复现。
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建数据集。
    x, y = make_synthetic_binary_dataset(num_samples=num_samples, input_dim=input_dim)
    dataset = TensorDataset(x, y)

    # 划分训练集与验证集。
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 模型规模适中：不过宽不过深，适配当前任务。
    model = SmallMLP(input_dim=input_dim, hidden1=32, hidden2=16, dropout_p=0.2).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    # 使用带权重衰减的 AdamW 作为正则化优化器。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    # 保存最优验证集权重，用于早停回退。
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # 仅在训练阶段执行数据增强。
            xb = augment_batch(xb, noise_std=noise_std, feature_drop_prob=feature_drop_prob)

            # 标准优化步骤：前向、反向、参数更新。
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 带 min_delta 容忍区间的改进判断。
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        # 验证集连续 `patience` 轮无提升则提前停止。
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # 恢复验证集表现最好的模型权重。
    model.load_state_dict(best_state)
    return model, history, device


if __name__ == "__main__":
    # 运行完整的正则化训练示例。
    model, history, device = train_with_regularization()
    checkpoint_path = save_checkpoint(model, history)
    curve_path = plot_history(history)
    reloaded_model, _ = load_checkpoint(checkpoint_path, device=device)

    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Best val loss: {min(history['val_loss']):.6f}")
    print(f"Best val acc:  {max(history['val_acc']):.4f}")
    print(f"Checkpoint saved: {checkpoint_path}")
    if curve_path is not None:
        print(f"Curve saved:      {curve_path}")
    print(f"Reloaded params:  {count_parameters(reloaded_model):,}")
