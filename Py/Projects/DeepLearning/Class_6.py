from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "Class_6"


def artifact_path(filename: str) -> str:
    """返回位于当前脚本目录下 artifacts/Class_6 中的绝对路径。"""

    return str(ARTIFACTS_DIR / filename)


# ------------------------------------------------------------
# 第 6 课：Deep Learning Limitations and New Frontiers
#
# 这一讲的主题和前几节课很不一样：
# 前面的课程大多在讲“怎样让模型学得更好”，而这一课会刻意讨论：
# 1. 深度网络为什么既强大，又可能出现令人不安的 failure mode；
# 2. 模型拥有很大容量时，为什么甚至可以记住随机标签；
# 3. 为什么看起来几乎不变的输入扰动，也可能让模型预测翻转；
# 4. 面对这些问题，我们有哪些常见的工程缓解思路。
#
# 为了保持和前几份脚本一致的“教学型、可直接运行”风格，这里仍然不用外部数据集，
# 而是构造两个尽量小但现象清晰的实验：
#
# 实验 A：容量与记忆化
# - 使用一个 2D 非线性分类任务；
# - 对比“真实标签训练”和“随机标签训练”；
# - 展示：模型甚至能把随机标签记住，但泛化会明显恶化。
#
# 实验 B：对抗样本与鲁棒性
# - 使用一个小型合成图形分类任务；
# - 先训练普通 CNN；
# - 再用 FGSM 生成对抗扰动；
# - 最后加入最基础的对抗训练，看鲁棒性是否改善。
#
# 目标不是追求 SOTA，而是把 Lecture 6 里最值得亲手感受的两个核心现象
# “memorization” 和 “adversarial vulnerability”
# 变成一份可以阅读、运行和导出结果的完整脚本。
# ------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """固定随机种子，尽量让实验结果可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """统计模型中的可训练参数量。"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_text_report(text: str, path: str) -> str:
    """把文本报告保存到磁盘。"""

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return str(file)


def moving_average(values: list[float], window: int = 20) -> list[float]:
    """返回滑动平均序列，便于观察训练趋势。"""

    if not values:
        return []

    result: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def _load_pillow_font(size: int = 16):
    """尽量加载支持中文的字体；若失败，则退回默认字体。"""

    try:
        from PIL import ImageFont
    except ImportError:
        return None

    candidate_fonts = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for font_path in candidate_fonts:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue

    return ImageFont.load_default()


def _ascii_fallback(text: str) -> str:
    """在极端字体缺失时，把文本降级为 ASCII，避免绘图阶段报错。"""

    fallback = text.encode("ascii", errors="ignore").decode("ascii")
    return fallback if fallback else "text"


def _safe_draw_text(draw, position: tuple[int, int], text: str, font, fill: str = "black") -> None:
    """安全绘制文本；若字体不支持中文，就自动退回 ASCII。"""

    try:
        draw.text(position, text, fill=fill, font=font)
    except UnicodeEncodeError:
        draw.text(position, _ascii_fallback(text), fill=fill, font=font)


def save_multi_curve_plot(
    series_map: dict[str, list[float]],
    path: str,
    title: str,
    y_label: str,
    series_labels: dict[str, str] | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    width: int = 960,
    height: int = 540,
) -> str | None:
    """用 Pillow 画多条曲线，避免额外依赖 matplotlib。"""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("未安装 Pillow，跳过曲线图导出。")
        return None

    valid_items = [(name, values) for name, values in series_map.items() if values]
    if not valid_items:
        return None

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_pillow_font(size=16)

    margin_left = 96
    margin_right = 32
    margin_top = 72
    margin_bottom = 84
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom

    title_font = _load_pillow_font(size=18)
    _safe_draw_text(draw, (margin_left, 18), title, font=title_font, fill="black")
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black", width=2)
    _safe_draw_text(draw, (24, plot_top - 2), y_label, font=font, fill="gray")
    _safe_draw_text(draw, (plot_right - 48, plot_bottom + 20), "epoch", font=font, fill="gray")

    all_values = [value for _, values in valid_items for value in values]
    vmin = min(all_values) if y_min is None else y_min
    vmax = max(all_values) if y_max is None else y_max
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1.0

    max_len = max(len(values) for _, values in valid_items)

    def project(idx: int, value: float, total_len: int) -> tuple[int, int]:
        x_ratio = idx / max(total_len - 1, 1)
        y_ratio = (value - vmin) / (vmax - vmin)
        x = plot_left + int(x_ratio * (plot_right - plot_left))
        y = plot_bottom - int(y_ratio * (plot_bottom - plot_top))
        return x, y

    colors = [
        (33, 150, 243),
        (244, 67, 54),
        (0, 150, 136),
        (255, 152, 0),
        (156, 39, 176),
        (121, 85, 72),
    ]

    # 先画几条水平参考线，让曲线更容易读数。
    tick_count = 5
    for tick_idx in range(tick_count):
        tick_ratio = tick_idx / max(tick_count - 1, 1)
        y_value = vmin + tick_ratio * (vmax - vmin)
        y = plot_bottom - int(tick_ratio * (plot_bottom - plot_top))
        draw.line([(plot_left, y), (plot_right, y)], fill=(232, 232, 232), width=1)
        _safe_draw_text(draw, (26, y - 8), f"{y_value:.2f}", font=font, fill="gray")

    legend_x = plot_left + 12
    legend_y = plot_top + 12
    legend_width = 250
    legend_height = 16 + 24 * len(valid_items)
    draw.rectangle(
        [legend_x - 8, legend_y - 6, legend_x - 8 + legend_width, legend_y - 6 + legend_height],
        fill=(250, 250, 250),
        outline=(220, 220, 220),
    )

    for idx, (name, values) in enumerate(valid_items):
        color = colors[idx % len(colors)]
        points = [project(i, v, len(values)) for i, v in enumerate(values)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        else:
            draw.line(points, fill=color, width=3)

        draw.rectangle([legend_x, legend_y + idx * 24, legend_x + 18, legend_y + 12 + idx * 24], fill=color)
        label_text = series_labels.get(name, name) if series_labels is not None else name
        _safe_draw_text(draw, (legend_x + 26, legend_y - 5 + idx * 24), label_text, font=font, fill="black")

    _safe_draw_text(draw, (plot_left, plot_bottom + 20), "0", font=font, fill="gray")
    _safe_draw_text(draw, (plot_right - 32, plot_bottom + 20), str(max_len), font=font, fill="gray")

    img.save(file)
    return str(file)


def save_scatter_plot(
    points: torch.Tensor,
    labels: torch.Tensor,
    path: str,
    title: str,
    decision_model: nn.Module | None = None,
    device: torch.device | None = None,
    width: int = 720,
    height: int = 720,
) -> str | None:
    """
    绘制 2D 点集散点图。

    如果提供了 decision_model，就顺便把当前模型的决策边界画出来。
    这样我们不只知道精度，还能更直观看到模型“把空间切成了什么样子”。
    """

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("未安装 Pillow，跳过散点图导出。")
        return None

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)

    font = _load_pillow_font(size=16)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    margin_left = 60
    margin_right = 25
    margin_top = 45
    margin_bottom = 55
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom

    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black", width=2)
    _safe_draw_text(draw, (margin_left, 12), title, font=font, fill="black")

    x_values = points[:, 0]
    y_values = points[:, 1]
    x_min = float(x_values.min().item()) - 0.25
    x_max = float(x_values.max().item()) + 0.25
    y_min = float(y_values.min().item()) - 0.25
    y_max = float(y_values.max().item()) + 0.25

    def world_to_pixel(x: float, y: float) -> tuple[int, int]:
        x_ratio = (x - x_min) / max(x_max - x_min, 1e-6)
        y_ratio = (y - y_min) / max(y_max - y_min, 1e-6)
        px = plot_left + int(x_ratio * (plot_right - plot_left))
        py = plot_bottom - int(y_ratio * (plot_bottom - plot_top))
        return px, py

    # 如果有模型，就先在背景上铺一层决策区域色块。
    if decision_model is not None and device is not None:
        decision_model.eval()
        grid_size = 150
        xs = torch.linspace(x_min, x_max, steps=grid_size)
        ys = torch.linspace(y_min, y_max, steps=grid_size)
        mesh_x, mesh_y = torch.meshgrid(xs, ys, indexing="xy")
        grid_points = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=1).to(device)
        with torch.no_grad():
            pred = decision_model(grid_points).argmax(dim=1).cpu().view(grid_size, grid_size)

        class_colors = {
            0: (230, 245, 255),
            1: (255, 235, 235),
        }
        for ix in range(grid_size - 1):
            for iy in range(grid_size - 1):
                px0, py0 = world_to_pixel(float(xs[ix].item()), float(ys[iy].item()))
                px1, py1 = world_to_pixel(float(xs[ix + 1].item()), float(ys[iy + 1].item()))
                color = class_colors[int(pred[iy, ix].item())]
                draw.rectangle([px0, py1, px1, py0], fill=color)

        draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="black", width=2)

    point_colors = {
        0: (33, 150, 243),
        1: (244, 67, 54),
    }
    for point, label in zip(points, labels):
        px, py = world_to_pixel(float(point[0].item()), float(point[1].item()))
        color = point_colors[int(label.item())]
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color, outline="black")

    _safe_draw_text(draw, (plot_left, plot_bottom + 10), f"x:[{x_min:.2f}, {x_max:.2f}]", font=font, fill="gray")
    _safe_draw_text(draw, (plot_left + 250, plot_bottom + 10), f"y:[{y_min:.2f}, {y_max:.2f}]", font=font, fill="gray")
    img.save(file)
    return str(file)


def save_image_grid(
    image_rows: list[list[torch.Tensor]],
    path: str,
    title: str,
    cell_size: int = 96,
    row_labels: list[str] | None = None,
) -> str | None:
    """
    保存图像网格。

    image_rows 的每一行都应包含若干张形状为 [C, H, W] 的张量。
    这里主要用于展示 clean / perturbation / adversarial 的对比。
    """

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("未安装 Pillow，跳过图像网格导出。")
        return None

    if not image_rows or not image_rows[0]:
        return None

    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(image_rows)
    n_cols = len(image_rows[0])
    font = _load_pillow_font(size=15)

    left_margin = 100 if row_labels else 22
    top_margin = 52
    right_margin = 20
    bottom_margin = 20
    pad = 10

    width = left_margin + n_cols * cell_size + (n_cols - 1) * pad + right_margin
    height = top_margin + n_rows * cell_size + (n_rows - 1) * pad + bottom_margin
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    _safe_draw_text(draw, (left_margin, 14), title, font=font, fill="black")

    for row_idx, row in enumerate(image_rows):
        if row_labels:
            _safe_draw_text(
                draw,
                (18, top_margin + row_idx * (cell_size + pad) + cell_size // 2 - 8),
                row_labels[row_idx],
                font=font,
                fill="gray",
            )

        for col_idx, tensor in enumerate(row):
            tile = tensor.detach().cpu().float()
            if tile.dim() == 2:
                tile = tile.unsqueeze(0)
            if tile.size(0) == 1:
                tile = tile.repeat(3, 1, 1)
            tile = tile.clamp(0.0, 1.0)
            tile = (tile * 255.0).byte().permute(1, 2, 0).numpy()

            from PIL import Image

            tile_img = Image.fromarray(tile).resize((cell_size, cell_size), Image.NEAREST)
            x = left_margin + col_idx * (cell_size + pad)
            y = top_margin + row_idx * (cell_size + pad)
            img.paste(tile_img, (x, y))
            draw.rectangle([x, y, x + cell_size, y + cell_size], outline="black", width=1)

    img.save(file)
    return str(file)


def make_two_moons(n_samples: int, noise: float = 0.08) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构造一个不依赖 sklearn 的 two-moons 数据集。

    之所以选这个任务，是因为它足够简单：
    - 输入只有 2 维，便于可视化；
    - 但它又不是线性可分的，MLP 的表达能力能够明显体现出来。
    """

    num_first = n_samples // 2
    num_second = n_samples - num_first

    first_points: list[list[float]] = []
    second_points: list[list[float]] = []

    for _ in range(num_first):
        angle = random.random() * math.pi
        x = math.cos(angle)
        y = math.sin(angle)
        x += random.gauss(0.0, noise)
        y += random.gauss(0.0, noise)
        first_points.append([x, y])

    for _ in range(num_second):
        angle = random.random() * math.pi
        x = 1.0 - math.cos(angle)
        y = -math.sin(angle) + 0.5
        x += random.gauss(0.0, noise)
        y += random.gauss(0.0, noise)
        second_points.append([x, y])

    points = torch.tensor(first_points + second_points, dtype=torch.float32)
    labels = torch.tensor([0] * num_first + [1] * num_second, dtype=torch.long)
    return points, labels


class PointDataset(Dataset):
    """2D 点分类数据集。"""

    def __init__(self, points: torch.Tensor, labels: torch.Tensor):
        self.points = points.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return self.points.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.points[index], self.labels[index]


class PointMLP(nn.Module):
    """
    用于实验 A 的小型 MLP。

    故意把宽度设得不算太小，是为了让它具备足够的容量，
    从而可以在“随机标签”场景下表现出明显的记忆化行为。
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """评估分类模型，返回平均损失和准确率。"""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            total_loss += float(loss.item()) * inputs.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_count += inputs.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def train_point_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float = 1e-3,
    tag: str = "true",
) -> dict[str, list[float]]:
    """训练 point classifier，并记录训练/测试曲线。"""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item()) * inputs.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_count += inputs.size(0)

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)
        test_loss, test_acc = evaluate_classifier(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if epoch == 1 or epoch % 40 == 0 or epoch == epochs:
            print(
                f"[Mem-{tag}] epoch={epoch:03d} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | "
                f"test_loss={test_loss:.4f} | test_acc={test_acc:.3f}"
            )

    return history


def save_memorization_report(
    true_history: dict[str, list[float]],
    random_history: dict[str, list[float]],
    path: str,
) -> str:
    """把实验 A 的结论整理成文本报告。"""

    true_train_acc = true_history["train_acc"][-1]
    true_test_acc = true_history["test_acc"][-1]
    random_train_acc = random_history["train_acc"][-1]
    random_test_acc = random_history["test_acc"][-1]

    true_gap = true_train_acc - true_test_acc
    random_gap = random_train_acc - random_test_acc

    lines = [
        "实验 A：容量与记忆化报告",
        "",
        "一、最终指标",
        f"真实标签模型：train_acc={true_train_acc:.4f}, test_acc={true_test_acc:.4f}, gap={true_gap:.4f}",
        f"随机标签模型：train_acc={random_train_acc:.4f}, test_acc={random_test_acc:.4f}, gap={random_gap:.4f}",
        "",
        "二、现象解释",
        "1. 当标签是真实结构时，模型既能拟合训练集，也能在测试集上保持较好的泛化。",
        "2. 当训练标签被随机打乱后，模型依然可能把训练集记住，说明网络容量很强。",
        "3. 但随机标签不包含可泛化规律，因此测试精度会明显下降。",
        "4. 这正是 Lecture 6 中常强调的点：高容量不自动等于真正理解，记忆化和泛化是两回事。",
        "",
        "三、为什么这个 synthetic 实验会显得更容易",
        "1. 输入只有 2 维，问题结构非常简单，我们还能直接把决策边界画出来。",
        "2. 真实标签和测试集都来自同一个 two-moons 生成过程，因此分布偏移几乎没有。",
        "3. 噪声强度有限，类别形状非常规整，所以真实标签任务会显得异常干净。",
        "4. 随机标签模型没有达到绝对 100% 训练精度，主要是因为我们仍保留了有限训练轮数和常规优化设置；但 0.969 对 0.478 的结果已经足够体现“能记住却不能泛化”的核心现象。",
    ]
    return save_text_report("\n".join(lines), path)


def build_shape_prototype(label: int, image_size: int) -> torch.Tensor:
    """根据类别 id 构造一个基础图形模板。"""

    canvas = torch.zeros(1, image_size, image_size, dtype=torch.float32)

    if label == 0:
        canvas[:, 4:12, 4:12] = 1.0
    elif label == 1:
        canvas[:, 6:10, 2:14] = 1.0
    elif label == 2:
        canvas[:, 2:14, 6:10] = 1.0
    elif label == 3:
        for idx in range(3, image_size - 3):
            canvas[:, idx, idx] = 1.0
            canvas[:, idx, image_size - idx - 1] = 1.0
            if idx + 1 < image_size:
                canvas[:, idx + 1, idx] = 1.0
                canvas[:, idx, idx + 1] = 1.0
                canvas[:, idx + 1, image_size - idx - 1] = 1.0
    else:
        raise ValueError(f"Unsupported label: {label}")

    return canvas


def shift_image(image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """把图像做整数平移，超出边界的部分直接裁掉。"""

    shifted = torch.zeros_like(image)
    _, height, width = image.shape

    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx) if dx >= 0 else width
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy) if dy >= 0 else height
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[:, src_y0:src_y1, src_x0:src_x1]
    return shifted


def make_shape_image(label: int, image_size: int = 16) -> torch.Tensor:
    """
    生成一张带随机扰动的合成图形图像。

    数据仍然足够简单，但会加入：
    - 轻微平移；
    - 局部噪声；
    - 亮度抖动；
    这样模型不会只靠完全固定模板背下来。
    """

    image = build_shape_prototype(label, image_size)
    dx = random.randint(-2, 2)
    dy = random.randint(-2, 2)
    image = shift_image(image, dx=dx, dy=dy)

    brightness = 0.85 + 0.3 * random.random()
    image = image * brightness

    noise = 0.08 * torch.randn_like(image)
    image = (image + noise).clamp(0.0, 1.0)
    return image


class ShapeDataset(Dataset):
    """用于实验 B 的小型合成图像分类数据集。"""

    def __init__(self, num_samples: int, image_size: int = 16, seed: int = 0):
        # 为了让数据集在每次运行中稳定，我们在构造阶段单独固定一次随机种子，
        # 然后把样本预生成好；这样 DataLoader 的 shuffle 不会影响图像内容本身。
        rng_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        random.seed(seed)
        torch.manual_seed(seed)

        self.images: list[torch.Tensor] = []
        self.labels: list[int] = []
        for _ in range(num_samples):
            label = random.randrange(4)
            self.images.append(make_shape_image(label, image_size=image_size))
            self.labels.append(label)

        random.setstate(rng_state)
        torch.random.set_rng_state(torch_state)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], torch.tensor(self.labels[index], dtype=torch.long)


class TinyConvNet(nn.Module):
    """实验 B 使用的小型卷积网络。"""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 96),
            nn.ReLU(),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method.

    核心思想：
    - 先计算损失对输入的梯度；
    - 再沿着“让损失增大最快”的符号方向，给输入加上一个很小的扰动；
    - 扰动幅度由 epsilon 控制。
    """

    attack_images = images.detach().clone().requires_grad_(True)
    logits = model(attack_images)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, attack_images)[0]
    adv_images = attack_images + epsilon * grad.sign()
    return adv_images.detach().clamp(0.0, 1.0)


def evaluate_image_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack_epsilon: float | None = None,
) -> tuple[float, float]:
    """
    评估图像分类模型。

    如果 attack_epsilon 不为 None，就在评估时动态生成 FGSM 对抗样本，
    从而得到 adversarial accuracy。
    """

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if attack_epsilon is not None:
            adv_images = fgsm_attack(model, images, labels, epsilon=attack_epsilon)
            logits = model(adv_images)
        else:
            with torch.no_grad():
                logits = model(images)

        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += images.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def train_shape_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 14,
    lr: float = 1e-3,
    adversarial_epsilon: float | None = None,
    tag: str = "std",
) -> dict[str, list[float]]:
    """
    训练 shape classifier。

    adversarial_epsilon 为 None 时，是普通训练；
    否则执行最基础的对抗训练：
    - 先生成当前 batch 的对抗样本；
    - 然后把 clean loss 和 adversarial loss 做平均。
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = {
        "train_loss": [],
        "clean_acc": [],
        "adv_acc": [],
    }

    eval_adv_epsilon = 0.12

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            if adversarial_epsilon is None:
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            else:
                # 注意：为了生成攻击样本，需要单独做一次基于当前模型的输入梯度计算。
                # 这里先用当前模型找“最危险”的输入扰动方向，再重新前向 clean / adv 两份样本。
                adv_images = fgsm_attack(model, images, labels, epsilon=adversarial_epsilon)
                clean_logits = model(images)
                adv_logits = model(adv_images)
                clean_loss = F.cross_entropy(clean_logits, labels)
                adv_loss = F.cross_entropy(adv_logits, labels)
                loss = 0.5 * clean_loss + 0.5 * adv_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item()) * images.size(0)
            total_count += images.size(0)

        train_loss = total_loss / max(total_count, 1)
        _, clean_acc = evaluate_image_classifier(model, test_loader, device, attack_epsilon=None)
        _, adv_acc = evaluate_image_classifier(model, test_loader, device, attack_epsilon=eval_adv_epsilon)

        history["train_loss"].append(train_loss)
        history["clean_acc"].append(clean_acc)
        history["adv_acc"].append(adv_acc)

        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            print(
                f"[Adv-{tag}] epoch={epoch:02d} | "
                f"train_loss={train_loss:.4f} | clean_acc={clean_acc:.3f} | "
                f"fgsm@{eval_adv_epsilon:.2f}={adv_acc:.3f}"
            )

    return history


def save_adversarial_report(
    standard_metrics: dict[str, float],
    robust_metrics: dict[str, float],
    path: str,
) -> str:
    """输出实验 B 的结果总结。"""

    lines = [
        "实验 B：对抗样本与鲁棒性报告",
        "",
        "一、普通训练模型",
        f"clean_acc={standard_metrics['clean_acc']:.4f}",
        f"fgsm_acc_eps_0.12={standard_metrics['adv_acc_012']:.4f}",
        f"fgsm_acc_eps_0.18={standard_metrics['adv_acc_018']:.4f}",
        "",
        "二、对抗训练模型",
        f"clean_acc={robust_metrics['clean_acc']:.4f}",
        f"fgsm_acc_eps_0.12={robust_metrics['adv_acc_012']:.4f}",
        f"fgsm_acc_eps_0.18={robust_metrics['adv_acc_018']:.4f}",
        "",
        "三、现象解释",
        "1. 普通 CNN 在干净样本上通常表现很好，但在对抗扰动下精度会明显下降。",
        "2. 这说明模型学到的决策边界可能非常脆弱，小扰动也会把样本推过边界。",
        "3. 对抗训练会把训练过程的一部分计算预算用于提升局部鲁棒性。",
        "4. 代价是 clean accuracy 可能略有波动，但 adversarial accuracy 往往会提升。",
        "",
        "四、为什么这个 synthetic 实验结果会比真实视觉任务更理想化",
        "1. 图像只有 16x16，类别也只是 square / h_bar / v_bar / x_shape 这类规则图形，视觉复杂度远低于真实照片。",
        "2. 训练集和测试集来自同一合成规则，因此 clean accuracy 很容易做到接近 100%。",
        "3. 对抗训练在这个小任务上几乎没有 clean accuracy 代价，但在真实数据集上通常会存在更明显的精度-鲁棒性权衡。",
        "4. 所以这里更适合作为 Lecture 6 的“现象演示器”，而不是现实世界鲁棒性的最终结论。",
    ]
    return save_text_report("\n".join(lines), path)


def save_shape_checkpoint(model: nn.Module, filename: str) -> str:
    """保存 shape classifier 的 checkpoint。"""

    file = Path(artifact_path(filename))
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, file)
    return str(file)


def collect_adversarial_showcase(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float = 0.18,
    max_samples: int = 6,
) -> list[list[torch.Tensor]]:
    """
    收集几组会被 FGSM 扰动明显影响的样本，用于做图像展示。

    每组包含：
    - 原图
    - 扰动可视化
    - 对抗图
    """

    model.eval()
    rows: list[list[torch.Tensor]] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)

        with torch.no_grad():
            clean_pred = model(images).argmax(dim=1)
            adv_pred = model(adv_images).argmax(dim=1)

        for idx in range(images.size(0)):
            if clean_pred[idx].item() != adv_pred[idx].item():
                delta = adv_images[idx] - images[idx]
                delta_vis = ((delta / max(epsilon, 1e-6)) * 0.5 + 0.5).clamp(0.0, 1.0)
                rows.append([images[idx].cpu(), delta_vis.cpu(), adv_images[idx].cpu()])
                if len(rows) >= max_samples:
                    return rows

    # 如果没有足够多翻转样本，也至少返回前几张，保证网格图能导出。
    if not rows:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
            for idx in range(min(images.size(0), max_samples)):
                delta = adv_images[idx] - images[idx]
                delta_vis = ((delta / max(epsilon, 1e-6)) * 0.5 + 0.5).clamp(0.0, 1.0)
                rows.append([images[idx].cpu(), delta_vis.cpu(), adv_images[idx].cpu()])
            break

    return rows[:max_samples]


def run_memorization_lesson(device: torch.device) -> None:
    """运行实验 A：容量与记忆化。"""

    print("\n" + "=" * 72)
    print("第 6 课 A：容量很强的网络，也可能记住随机标签")

    set_seed(7)
    train_points, train_labels = make_two_moons(n_samples=64, noise=0.08)
    test_points, test_labels = make_two_moons(n_samples=320, noise=0.08)

    shuffled_labels = train_labels[torch.randperm(train_labels.size(0))]

    true_train_loader = DataLoader(PointDataset(train_points, train_labels), batch_size=64, shuffle=True)
    random_train_loader = DataLoader(PointDataset(train_points, shuffled_labels), batch_size=64, shuffle=True)
    test_loader = DataLoader(PointDataset(test_points, test_labels), batch_size=128, shuffle=False)

    true_model = PointMLP(hidden_dim=256).to(device)
    random_model = PointMLP(hidden_dim=256).to(device)

    print(f"真实标签模型参数量：{count_parameters(true_model)}")
    true_history = train_point_model(
        true_model,
        train_loader=true_train_loader,
        test_loader=test_loader,
        device=device,
        epochs=180,
        lr=2e-3,
        tag="true",
    )

    print(f"随机标签模型参数量：{count_parameters(random_model)}")
    random_history = train_point_model(
        random_model,
        train_loader=random_train_loader,
        test_loader=test_loader,
        device=device,
        epochs=1200,
        lr=2e-3,
        tag="rand",
    )

    train_curve_path = save_multi_curve_plot(
        {
            "true_train_acc": true_history["train_acc"],
            "rand_train_acc": random_history["train_acc"],
        },
        path=artifact_path("memorization_train_accuracy.png"),
        title="实验 A：训练集精度对比",
        y_label="accuracy",
        series_labels={
            "true_train_acc": "真实标签训练集精度",
            "rand_train_acc": "随机标签训练集精度",
        },
        y_min=0.35,
        y_max=1.02,
    )
    test_curve_path = save_multi_curve_plot(
        {
            "true_test_acc": true_history["test_acc"],
            "rand_test_acc": random_history["test_acc"],
        },
        path=artifact_path("memorization_accuracy_curves.png"),
        title="实验 A：测试集精度对比",
        y_label="accuracy",
        series_labels={
            "true_test_acc": "真实标签测试集精度",
            "rand_test_acc": "随机标签测试集精度",
        },
        y_min=0.35,
        y_max=1.02,
    )

    true_boundary = save_scatter_plot(
        train_points,
        train_labels,
        path=artifact_path("decision_boundary_true_labels.png"),
        title="真实标签训练后的决策边界",
        decision_model=true_model,
        device=device,
    )
    random_boundary = save_scatter_plot(
        train_points,
        shuffled_labels,
        path=artifact_path("decision_boundary_random_labels.png"),
        title="随机标签训练后的决策边界",
        decision_model=random_model,
        device=device,
    )
    report_path = save_memorization_report(
        true_history,
        random_history,
        path=artifact_path("memorization_report.txt"),
    )

    print(
        f"真实标签最终：train_acc={true_history['train_acc'][-1]:.3f}, "
        f"test_acc={true_history['test_acc'][-1]:.3f}"
    )
    print(
        f"随机标签最终：train_acc={random_history['train_acc'][-1]:.3f}, "
        f"test_acc={random_history['test_acc'][-1]:.3f}"
    )
    if train_curve_path is not None:
        print(f"记忆化训练集精度曲线已保存到：{train_curve_path}")
    if test_curve_path is not None:
        print(f"记忆化测试集精度曲线已保存到：{test_curve_path}")
    if true_boundary is not None:
        print(f"真实标签决策边界图已保存到：{true_boundary}")
    if random_boundary is not None:
        print(f"随机标签决策边界图已保存到：{random_boundary}")
    print(f"记忆化报告已保存到：{report_path}")


def run_adversarial_lesson(device: torch.device) -> None:
    """运行实验 B：对抗样本与鲁棒性。"""

    print("\n" + "=" * 72)
    print("第 6 课 B：对抗样本会暴露模型的脆弱性，对抗训练可提升鲁棒性")

    set_seed(21)
    train_dataset = ShapeDataset(num_samples=640, image_size=16, seed=21)
    test_dataset = ShapeDataset(num_samples=256, image_size=16, seed=121)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    standard_model = TinyConvNet(num_classes=4).to(device)
    robust_model = TinyConvNet(num_classes=4).to(device)

    print(f"普通 CNN 参数量：{count_parameters(standard_model)}")
    standard_history = train_shape_model(
        standard_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=14,
        lr=1e-3,
        adversarial_epsilon=None,
        tag="std",
    )

    print(f"对抗训练 CNN 参数量：{count_parameters(robust_model)}")
    robust_history = train_shape_model(
        robust_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=14,
        lr=1e-3,
        adversarial_epsilon=0.12,
        tag="robust",
    )

    clean_curve_path = save_multi_curve_plot(
        {
            "standard_clean_acc": standard_history["clean_acc"],
            "robust_clean_acc": robust_history["clean_acc"],
        },
        path=artifact_path("adversarial_clean_accuracy.png"),
        title="实验 B：干净样本精度对比",
        y_label="accuracy",
        series_labels={
            "standard_clean_acc": "普通训练 clean accuracy",
            "robust_clean_acc": "对抗训练 clean accuracy",
        },
        y_min=0.0,
        y_max=1.02,
    )
    curve_path = save_multi_curve_plot(
        {
            "standard_adv_acc": standard_history["adv_acc"],
            "robust_adv_acc": robust_history["adv_acc"],
        },
        path=artifact_path("adversarial_accuracy_curves.png"),
        title="实验 B：FGSM 样本精度对比",
        y_label="accuracy",
        series_labels={
            "standard_adv_acc": "普通训练 FGSM accuracy",
            "robust_adv_acc": "对抗训练 FGSM accuracy",
        },
        y_min=0.0,
        y_max=1.02,
    )

    standard_clean_loss, standard_clean_acc = evaluate_image_classifier(standard_model, test_loader, device, None)
    _, standard_adv_012 = evaluate_image_classifier(standard_model, test_loader, device, attack_epsilon=0.12)
    _, standard_adv_018 = evaluate_image_classifier(standard_model, test_loader, device, attack_epsilon=0.18)

    robust_clean_loss, robust_clean_acc = evaluate_image_classifier(robust_model, test_loader, device, None)
    _, robust_adv_012 = evaluate_image_classifier(robust_model, test_loader, device, attack_epsilon=0.12)
    _, robust_adv_018 = evaluate_image_classifier(robust_model, test_loader, device, attack_epsilon=0.18)

    report_path = save_adversarial_report(
        standard_metrics={
            "clean_loss": standard_clean_loss,
            "clean_acc": standard_clean_acc,
            "adv_acc_012": standard_adv_012,
            "adv_acc_018": standard_adv_018,
        },
        robust_metrics={
            "clean_loss": robust_clean_loss,
            "clean_acc": robust_clean_acc,
            "adv_acc_012": robust_adv_012,
            "adv_acc_018": robust_adv_018,
        },
        path=artifact_path("adversarial_report.txt"),
    )

    showcase_rows = collect_adversarial_showcase(standard_model, test_loader, device, epsilon=0.18, max_samples=6)
    showcase_path = save_image_grid(
        showcase_rows,
        path=artifact_path("adversarial_showcase.png"),
        title="普通 CNN 在 FGSM 攻击下的 clean / perturbation / adversarial",
        row_labels=[f"sample_{idx + 1}" for idx in range(len(showcase_rows))],
    )

    std_ckpt = save_shape_checkpoint(standard_model, "shape_classifier_standard.pt")
    robust_ckpt = save_shape_checkpoint(robust_model, "shape_classifier_robust.pt")

    print(
        f"普通模型：clean_loss={standard_clean_loss:.4f}, clean_acc={standard_clean_acc:.3f}, "
        f"fgsm@0.12={standard_adv_012:.3f}, fgsm@0.18={standard_adv_018:.3f}"
    )
    print(
        f"对抗训练模型：clean_loss={robust_clean_loss:.4f}, clean_acc={robust_clean_acc:.3f}, "
        f"fgsm@0.12={robust_adv_012:.3f}, fgsm@0.18={robust_adv_018:.3f}"
    )
    if clean_curve_path is not None:
        print(f"干净样本精度曲线已保存到：{clean_curve_path}")
    if curve_path is not None:
        print(f"对抗鲁棒性曲线已保存到：{curve_path}")
    if showcase_path is not None:
        print(f"对抗样本展示图已保存到：{showcase_path}")
    print(f"普通模型 checkpoint 已保存到：{std_ckpt}")
    print(f"对抗训练模型 checkpoint 已保存到：{robust_ckpt}")
    print(f"对抗鲁棒性报告已保存到：{report_path}")


def main() -> None:
    """程序入口。"""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备：{device}")
    print("将依次运行：实验 A（记忆化） 和 实验 B（对抗鲁棒性）")

    run_memorization_lesson(device)
    run_adversarial_lesson(device)


if __name__ == "__main__":
    main()
