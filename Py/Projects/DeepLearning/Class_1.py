import copy
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# 本文件演示一个完整的小型深度学习训练流程：
# 1. 构造合成二分类数据集。
# 2. 使用 MLP 模型进行二分类预测。
# 3. 加入数据增强、Dropout、AdamW 权重衰减和早停机制来降低过拟合风险。
# 4. 保存/加载最佳模型检查点，并把训练曲线画成图片。


def make_synthetic_binary_dataset(num_samples=4000, input_dim=20):
    """生成一个简单的二分类合成数据集。

    num_samples 表示样本数量，input_dim 表示每个样本有多少个特征。
    这个函数会先随机生成输入特征 x，再用一个隐藏的线性规则生成标签 y。
    """

    # 随机生成特征。
    # x 的形状是 [num_samples, input_dim]，每一行代表一个样本。
    x = torch.randn(num_samples, input_dim)
    # 使用隐藏线性规则并叠加噪声，模拟真实数据分布。
    # true_w 相当于“真实但未知”的分类规则，模型训练时并不知道它。
    true_w = torch.randn(input_dim, 1)
    # logits 是线性打分；额外的随机噪声让任务更接近真实数据，不至于完全可分。
    logits = x @ true_w + 0.25 * torch.randn(num_samples, 1)
    # 将概率转换为二值标签 {0, 1}。
    # sigmoid(logits) 会把任意实数压缩到 0~1，可理解为属于正类的概率。
    y = (torch.sigmoid(logits) > 0.5).float()
    return x, y


def augment_batch(x, noise_std=0.05, feature_drop_prob=0.05):
    """对表格类张量做轻量级数据增强。

    数据增强只在训练阶段使用，用来让模型不要过度依赖某些固定输入模式。
    noise_std 控制高斯噪声强度，feature_drop_prob 控制随机丢弃特征的概率。
    """

    # 添加高斯噪声。
    # torch.randn_like(x) 会生成和 x 同形状的标准正态噪声。
    noise = torch.randn_like(x) * noise_std
    # 随机丢弃部分特征（输入层正则化）。
    # keep_mask 中 1 表示保留该特征，0 表示把该特征置为 0。
    keep_mask = (torch.rand_like(x) > feature_drop_prob).float()
    return (x + noise) * keep_mask


class SmallMLP(nn.Module):
    """适用于小中型玩具任务的紧凑 MLP 模型。

    MLP（Multi-Layer Perceptron，多层感知机）适合处理这种固定长度的表格特征。
    这里输出维度为 1，对应二分类任务中的一个 logit。
    """

    def __init__(self, input_dim, hidden1=32, hidden2=16, dropout_p=0.2):
        super().__init__()
        # nn.Sequential 会按顺序执行其中的层，适合搭建简单前馈网络。
        self.net = nn.Sequential(
            # 第一层把 input_dim 个原始特征映射到 hidden1 维隐藏表示。
            nn.Linear(input_dim, hidden1),
            # ReLU 引入非线性，否则多层线性层叠加后仍然等价于一层线性模型。
            nn.ReLU(),
            # Dropout 训练时随机置零一部分隐藏单元，是常用正则化手段。
            nn.Dropout(dropout_p),
            # 第二层继续压缩/变换隐藏表示。
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            # 最后一层输出一个 logit，不在模型里加 sigmoid，因为 BCEWithLogitsLoss 会内部处理。
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        """执行前向传播，输入形状通常为 [batch_size, input_dim]。"""

        return self.net(x)


def count_parameters(model):
    # 仅统计可训练参数数量。
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_model_config(model):
    """从 SmallMLP 实例中提取结构配置。

    保存模型时不能只保存权重，还要保存模型结构参数。
    这样加载检查点时才能重新创建出完全相同结构的 SmallMLP。
    """

    # 这里通过 Sequential 中固定的层索引读取输入维度、隐藏层维度和 Dropout 概率。
    return {
        "input_dim": model.net[0].in_features,
        "hidden1": model.net[0].out_features,
        "hidden2": model.net[3].out_features,
        "dropout_p": model.net[2].p,
    }


def save_checkpoint(model, history, checkpoint_path="artifacts/best_mlp_checkpoint.pt"):
    """保存模型权重、结构配置和训练历史。

    checkpoint 是一个字典，包含恢复模型和分析训练过程所需的信息。
    """

    checkpoint = {
        # state_dict 保存的是每一层的参数张量，不直接保存整个 Python 对象。
        "model_state_dict": model.state_dict(),
        # model_config 用于加载时重建模型结构。
        "model_config": extract_model_config(model),
        # history 记录训练/验证指标，便于之后画图或分析。
        "history": history,
    }
    checkpoint_file = Path(checkpoint_path)
    # 如果 artifacts 目录不存在，先自动创建。
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_file)
    return str(checkpoint_file)


def load_checkpoint(checkpoint_path, device=None):
    """从检查点加载模型，并恢复对应结构。

    device 用来控制模型加载到 CPU 还是 GPU。
    如果没有传入 device，就优先使用 CUDA，否则使用 CPU。
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # map_location 确保即使保存时在 GPU，加载时也可以映射到当前可用设备。
    # weights_only=True 更偏向只加载张量和基础对象，减少加载任意对象的安全风险。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["model_config"]

    # 根据保存的结构配置重新创建模型。
    model = SmallMLP(
        input_dim=config["input_dim"],
        hidden1=config["hidden1"],
        hidden2=config["hidden2"],
        dropout_p=config["dropout_p"],
    ).to(device)
    # 再把保存的参数权重加载回模型。
    model.load_state_dict(checkpoint["model_state_dict"])
    # 加载后切换到评估模式，避免 Dropout 在推理时继续随机失活。
    model.eval()
    return model, checkpoint.get("history")


def plot_history(history, save_path="artifacts/training_curves.png"):
    """使用 Pillow 绘制并保存训练/验证曲线。

    为了避免额外依赖 matplotlib，这里直接用 Pillow 画简单折线图。
    如果环境没有安装 Pillow，就跳过绘图，不影响训练和模型保存。
    """

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

    # 画布和布局参数：左边画 Loss，右边画验证准确率。
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
        """把一串指标数值转换为图像坐标点。"""

        if len(values) == 1:
            return [(x0, y0 + h // 2)]
        denom = max(y_max - y_min, 1e-12)
        points = []
        for i, v in enumerate(values):
            # x 坐标按 epoch 顺序均匀分布。
            x = x0 + int(i * w / (len(values) - 1))
            # 图像坐标系 y 轴向下，所以数值越大，y 坐标要越靠上。
            y = y0 + h - int((v - y_min) * h / denom)
            points.append((x, y))
        return points

    def draw_panel(x0, title, series, y_min, y_max):
        """绘制一个子图面板，包括边框、标题、折线和图例。"""

        y0 = panel_top
        draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], outline="black", width=1)
        draw.text((x0, y0 - 18), title, fill="black", font=font)

        for idx, (name, values, color) in enumerate(series):
            pts = to_points(values, x0 + 8, y0 + 8, panel_w - 16, panel_h - 16, y_min, y_max)
            if len(pts) > 1:
                draw.line(pts, fill=color, width=2)
            # 在面板下方写出每条曲线的名称，充当简易图例。
            draw.text((x0 + 8, y0 + panel_h + 8 + 12 * idx), name, fill=color, font=font)

        draw.text((x0 + panel_w - 120, y0 + panel_h + 8), f"min={y_min:.4f}", fill="gray", font=font)
        draw.text((x0 + panel_w - 120, y0 + panel_h + 20), f"max={y_max:.4f}", fill="gray", font=font)

    # Loss 面板需要同时包含训练损失和验证损失，因此先合并计算 y 轴范围。
    loss_all = train_loss + val_loss
    loss_min = min(loss_all)
    loss_max = max(loss_all)
    if abs(loss_max - loss_min) < 1e-12:
        # 如果所有 loss 几乎相同，手动扩大一点范围，避免除以 0。
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
    # 保存前确保输出目录存在。
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    img.save(plot_file)
    return str(plot_file)


def evaluate(model, loader, loss_fn, device):
    """在验证集或测试集上评估模型，返回平均损失和准确率。"""

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

            # BCEWithLogitsLoss 训练时直接使用 logits；
            # 计算准确率时再手动 sigmoid，把 logits 转成 0~1 的概率。
            probs = torch.sigmoid(logits)
            # 二分类阈值设为 0.5，概率大于等于 0.5 判为正类 1，否则判为 0。
            preds = (probs >= 0.5).float()

            # loss.item() 是当前 batch 的平均损失，乘 batch 大小后累加，最后再除以总样本数。
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
    """包含数据增强 + AdamW + 早停机制的训练流程。

    返回训练好的模型、训练历史和当前使用的设备。
    这个函数把数据准备、模型创建、训练、验证和早停都组织在一起。
    """

    # 固定随机种子，保证实验可复现。
    torch.manual_seed(seed)
    # 如果本机有可用 GPU，就优先使用 CUDA；否则使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建数据集。
    x, y = make_synthetic_binary_dataset(num_samples=num_samples, input_dim=input_dim)
    dataset = TensorDataset(x, y)

    # 划分训练集与验证集。
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # 训练集打乱顺序，验证集不打乱，保证评估更稳定。
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 模型规模适中：不过宽不过深，适配当前任务。
    model = SmallMLP(input_dim=input_dim, hidden1=32, hidden2=16, dropout_p=0.2).to(device)
    # BCEWithLogitsLoss = sigmoid + binary cross entropy，数值上比手动 sigmoid 后再算 BCE 更稳定。
    loss_fn = nn.BCEWithLogitsLoss()
    # 使用带权重衰减的 AdamW 作为正则化优化器。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # history 用于记录每轮训练后的指标，后面可以保存或绘制曲线。
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    # 保存最优验证集权重，用于早停回退。
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    # wait 记录验证损失连续多少轮没有明显提升。
    wait = 0

    for epoch in range(1, max_epochs + 1):
        # 开启训练模式，Dropout 会在训练时生效。
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # 仅在训练阶段执行数据增强。
            xb = augment_batch(xb, noise_std=noise_std, feature_drop_prob=feature_drop_prob)

            # 标准优化步骤：前向、反向、参数更新。
            # zero_grad 清空上一轮 batch 的梯度，避免梯度累积。
            optimizer.zero_grad()
            # 前向传播得到 logits，形状为 [batch_size, 1]。
            logits = model(xb)
            # 和真实标签 yb 计算二分类损失。
            loss = loss_fn(logits, yb)
            # 反向传播计算每个参数的梯度。
            loss.backward()
            # 根据梯度更新模型参数。
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
            # deepcopy 很重要：如果直接引用 state_dict，后续训练会继续修改里面的张量。
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
    # 保存最优模型检查点，包含结构配置、权重和训练历史。
    checkpoint_path = save_checkpoint(model, history)
    # 保存训练曲线图片；如果 Pillow 未安装，curve_path 会是 None。
    curve_path = plot_history(history)
    # 重新加载检查点，验证保存/加载流程可用。
    reloaded_model, _ = load_checkpoint(checkpoint_path, device=device)

    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Best val loss: {min(history['val_loss']):.6f}")
    print(f"Best val acc:  {max(history['val_acc']):.4f}")
    print(f"Checkpoint saved: {checkpoint_path}")
    if curve_path is not None:
        print(f"Curve saved:      {curve_path}")
    print(f"Reloaded params:  {count_parameters(reloaded_model):,}")
