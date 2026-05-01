import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# 本文件对应 MIT 6.S191 第 2 课“深度序列建模”的教学演示：
# 1. 说明序列建模的任务类型、设计准则和 BPTT 的梯度问题。
# 2. 分别训练 RNN、LSTM、GRU 和 Transformer 做字符级语言建模。
# 3. 保存损失曲线、生成文本、模型对比、checkpoint 和注意力热力图。

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "Class_2"


def artifact_path(filename: str) -> Path:
    """返回本课统一的产物路径，并确保目录存在。"""

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / filename


class Utf8Tee:
    """同时把输出写到终端和 UTF-8 日志文件。"""

    def __init__(self, console_stream, file_path: Path):
        self.console_stream = console_stream
        self.log_file = file_path.open("w", encoding="utf-8")

    def write(self, text: str) -> int:
        self.console_stream.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.console_stream.flush()
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


def set_seed(seed: int = 42) -> None:
    """固定随机种子，尽量保证每次运行结果一致。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_sequence_modeling_tasks() -> None:
    """打印常见的序列建模任务范式。"""

    task_map = {
        "One-to-One": "普通二分类",
        "Many-to-One": "情感分类 / 时间序列预测",
        "One-to-Many": "图像描述 / 文本生成",
        "Many-to-Many": "机器翻译 / 语音识别",
    }
    print("序列建模应用场景：")
    for mapping, task in task_map.items():
        print(f"- {mapping}: {task}")


def describe_sequence_design_principles() -> None:
    """打印课件中的四大设计准则。"""

    print("序列模型的四大设计准则：")
    print("- 处理变长序列")
    print("- 跟踪长依赖")
    print("- 保持顺序信息")
    print("- 参数共享")


def demonstrate_gradient_flow() -> None:
    """用简单连乘直观展示梯度消失和梯度爆炸。"""

    small_gain = 0.8
    large_gain = 1.2
    steps = 20
    print("BPTT 梯度现象演示：")
    print(f"- 连续乘以 {small_gain:.1f}，经过 {steps} 步后约为 {small_gain**steps:.6f}")
    print(f"- 连续乘以 {large_gain:.1f}，经过 {steps} 步后约为 {large_gain**steps:.6f}")


def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """根据语料构建字符级词表。"""

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """把字符串编码成 token id 张量。"""

    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(token_ids: torch.Tensor, itos: dict[int, str]) -> str:
    """把 token id 张量解码回字符串。"""

    return "".join(itos[int(i)] for i in token_ids)


def make_lm_dataset(token_ids: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """把长文本切成“根据前文预测下一个字符”的训练样本。"""

    x_list, y_list = [], []
    for i in range(len(token_ids) - block_size):
        x_list.append(token_ids[i : i + block_size])
        y_list.append(token_ids[i + 1 : i + block_size + 1])
    return torch.stack(x_list), torch.stack(y_list)


class SimpleRNNLM(nn.Module):
    """最基础的字符级 RNN 语言模型。"""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        hidden_states, _ = self.rnn(x)
        return self.head(hidden_states)


class LSTMLM(nn.Module):
    """使用门控机制缓解长依赖问题的 LSTM 语言模型。"""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        hidden_states, _ = self.lstm(x)
        return self.head(hidden_states)


class GRULM(nn.Module):
    """使用更新门和重置门的 GRU 语言模型。"""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        hidden_states, _ = self.gru(x)
        return self.head(hidden_states)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力，展示 Transformer 的核心运算。"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        attn_mask = None
        if causal:
            attn_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=x.device),
                diagonal=1,
            )

        attended, attn_weights = self.attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return attended, attn_weights


class TinyTransformerBlock(nn.Module):
    """极简 Transformer Block。"""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(x, causal=True)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x, attn_weights


class TinyTransformerLM(nn.Module):
    """简化版字符级 Transformer 语言模型。"""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(embed_dim, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t = x.shape
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(bsz, t)
        h = self.token_embedding(x) + self.pos_embedding(positions)
        for block in self.blocks:
            h, _ = block(h)
        return self.head(h)

    @torch.no_grad()
    def get_attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """返回每一层的注意力权重。"""

        bsz, t = x.shape
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(bsz, t)
        h = self.token_embedding(x) + self.pos_embedding(positions)
        weights_per_layer: list[torch.Tensor] = []
        for block in self.blocks:
            h, weights = block(h)
            weights_per_layer.append(weights)
        return weights_per_layer


@dataclass
class TrainConfig:
    """训练超参数配置。"""

    epochs: int = 60
    lr: float = 3e-3
    batch_size: int = 128
    grad_clip: float = 1.0
    val_ratio: float = 0.2


def split_train_val(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    val_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按固定随机种子拆分训练集和验证集。"""

    total = x_data.size(0)
    val_size = max(1, int(total * val_ratio))
    indices = torch.randperm(total)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return x_data[train_indices], y_data[train_indices], x_data[val_indices], y_data[val_indices]


@torch.no_grad()
def evaluate_language_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """计算验证集平均交叉熵损失。"""

    model.eval()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    model_name: str,
    model_config: dict[str, int | float],
    best_val_loss: float,
) -> None:
    """保存模型权重和基础元信息。"""

    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "config": model_config,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    fill: tuple[int, int, int],
    width: int = 2,
    dashed: bool = False,
) -> None:
    """绘制实线或虚线折线。"""

    if len(points) < 2:
        return
    if not dashed:
        draw.line(points, fill=fill, width=width)
        return
    for idx in range(len(points) - 1):
        if idx % 2 == 0:
            draw.line([points[idx], points[idx + 1]], fill=fill, width=width)


def train_language_model(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    tag: str,
    checkpoint_path: Path,
    model_config: dict[str, int | float],
) -> dict[str, float | list[float]]:
    """统一训练流程，返回训练/验证曲线与耗时。"""

    model.to(device)
    train_x, train_y, val_x, val_y = split_train_val(x_data, y_data, cfg.val_ratio)
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    train_losses: list[float] = []
    val_losses: list[float] = []
    start_time = time.perf_counter()
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = evaluate_language_model(model, val_loader, device)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, checkpoint_path, tag, model_config, best_val_loss)

        if epoch == 1 or epoch % 15 == 0:
            print(
                f"[{tag}] 轮次 {epoch:3d} | "
                f"训练损失={avg_train_loss:.4f} | 验证损失={avg_val_loss:.4f}"
            )

    elapsed = time.perf_counter() - start_time
    return {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss,
        "train_seconds": elapsed,
        "avg_epoch_seconds": elapsed / cfg.epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


@torch.no_grad()
def generate_text(
    model: nn.Module,
    seed_text: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    block_size: int,
    new_tokens: int = 80,
    temperature: float = 0.9,
) -> str:
    """使用自回归方式生成文本。"""

    model.eval()
    device = next(model.parameters()).device
    context = torch.tensor([[stoi[ch] for ch in seed_text]], dtype=torch.long, device=device)

    for _ in range(new_tokens):
        x_cond = context[:, -block_size:]
        logits = model(x_cond)
        next_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)

    return decode(context[0].cpu(), itos)


def save_loss_curves(histories: dict[str, dict[str, list[float]]], save_path: Path) -> None:
    """保存训练/验证损失曲线，并标注每条曲线最终损失。"""

    width, height = 1280, 700
    left, right, top, bottom = 100, 250, 55, 85
    plot_width = width - left - right
    plot_height = height - top - bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = {
        "RNN": (231, 76, 60),
        "LSTM": (46, 134, 193),
        "GRU": (155, 89, 182),
        "Transformer": (39, 174, 96),
    }

    all_losses = []
    for item in histories.values():
        all_losses.extend(item["train"])
        all_losses.extend(item["val"])
    max_epochs = max(len(item["train"]) for item in histories.values())
    raw_min_loss = min(all_losses)
    raw_max_loss = max(all_losses)
    padding = max((raw_max_loss - raw_min_loss) * 0.08, 0.01)
    min_loss = max(0.0, raw_min_loss - padding)
    max_loss = raw_max_loss + padding
    loss_span = max(max_loss - min_loss, 1e-6)

    draw.rectangle([left, top, left + plot_width, top + plot_height], outline="black", width=2)
    draw.text((left, 18), "Class 2 Training and Validation Loss Curves", fill="black")
    draw.text((left + plot_width // 2 - 20, height - 32), "epoch", fill="black")
    draw.text((22, top + plot_height // 2), "loss", fill="black")

    y_tick_count = 7
    for i in range(y_tick_count):
        y = top + int(plot_height * i / (y_tick_count - 1))
        draw.line([(left, y), (left + plot_width, y)], fill=(220, 220, 220), width=1)
        tick_loss = max_loss - (max_loss - min_loss) * i / (y_tick_count - 1)
        draw.text((32, y - 8), f"{tick_loss:.2f}", fill="black")

    x_tick_count = 5
    for i in range(x_tick_count):
        x = left + int(plot_width * i / (x_tick_count - 1))
        draw.line([(x, top), (x, top + plot_height)], fill=(235, 235, 235), width=1)
        tick_epoch = 1 + int((max_epochs - 1) * i / (x_tick_count - 1))
        draw.text((x - 8, top + plot_height + 10), f"{tick_epoch}", fill="black")

    legend_x = left + plot_width + 20
    legend_y = top + 16
    for model_name, curves in histories.items():
        color = colors.get(model_name, (0, 0, 0))
        for curve_kind, dashed in (("train", False), ("val", True)):
            losses = curves[curve_kind]
            points: list[tuple[int, int]] = []
            for idx, loss in enumerate(losses):
                x = left + int(plot_width * idx / max(max_epochs - 1, 1))
                y_ratio = (loss - min_loss) / loss_span
                y = top + plot_height - int(plot_height * y_ratio)
                points.append((x, y))
            draw_polyline(draw, points, fill=color, width=3 if not dashed else 2, dashed=dashed)
            final_x, final_y = points[-1]
            draw.ellipse((final_x - 4, final_y - 4, final_x + 4, final_y + 4), fill=color, outline=color)
            if curve_kind == "val":
                label_x = final_x + 10
                label_y = final_y - 8
                if label_x > left + plot_width - 120:
                    label_x = left + plot_width - 120
                draw.text((label_x, label_y), f"{losses[-1]:.4f}", fill=color)

        draw.line([(legend_x, legend_y + 8), (legend_x + 24, legend_y + 8)], fill=color, width=4)
        draw.text((legend_x + 32, legend_y), f"{model_name} train", fill="black")
        draw.line([(legend_x, legend_y + 28), (legend_x + 24, legend_y + 28)], fill=color, width=2)
        draw.line([(legend_x + 8, legend_y + 28), (legend_x + 16, legend_y + 28)], fill="white", width=2)
        draw.text((legend_x + 32, legend_y + 20), f"{model_name} val", fill="black")
        draw.text((legend_x + 150, legend_y + 20), f"val end={curves['val'][-1]:.4f}", fill=color)
        legend_y += 52

    image.save(save_path)


def save_generation_report(
    samples: dict[str, str],
    dependency_results: dict[str, dict[str, str]],
    save_path: Path,
) -> None:
    """保存各模型的生成文本和长依赖测试结果。"""

    lines = ["Class 2 生成文本报告", "=" * 40, ""]
    for model_name, text in samples.items():
        lines.append(f"{model_name} 生成示例：")
        lines.append(text)
        lines.append("")

    lines.append("长依赖测试：")
    lines.append("")
    for prompt_name, outputs in dependency_results.items():
        lines.append(f"{prompt_name}：")
        for model_name, text in outputs.items():
            lines.append(f"- {model_name}: {text}")
        lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")


def save_model_comparison(
    results: dict[str, dict[str, float | str]],
    dependency_results: dict[str, dict[str, str]],
    head_comparison: list[dict[str, float]],
    save_path: Path,
) -> None:
    """保存模型对比报告。"""

    lines = [
        "Class 2 序列模型对比报告",
        "=" * 40,
        "",
        "整体结果：",
    ]
    for model_name, info in results.items():
        lines.append(
            f"- {model_name}: 参数量={int(info['params'])}, "
            f"训练时间={float(info['train_seconds']):.2f}s, "
            f"平均每轮耗时={float(info['avg_epoch_seconds']):.3f}s, "
            f"最终训练损失={float(info['final_train_loss']):.4f}, "
            f"最终验证损失={float(info['final_val_loss']):.4f}, "
            f"最佳验证损失={float(info['best_val_loss']):.4f}"
        )

    lines.extend(
        [
            "",
            "门控模型与注意力模型对比：",
            f"- LSTM 最佳验证损失：{float(results['LSTM']['best_val_loss']):.4f}",
            f"- GRU 最佳验证损失：{float(results['GRU']['best_val_loss']):.4f}",
            f"- Transformer 最佳验证损失：{float(results['Transformer']['best_val_loss']):.4f}",
            f"- LSTM 训练时间：{float(results['LSTM']['train_seconds']):.2f}s",
            f"- GRU 训练时间：{float(results['GRU']['train_seconds']):.2f}s",
            f"- Transformer 训练时间：{float(results['Transformer']['train_seconds']):.2f}s",
            "",
            "长依赖测试摘要：",
        ]
    )
    lines.append("Transformer 结构：")
    lines.append(f"- 层数：{int(results['Transformer']['num_layers'])}")
    lines.append(f"- 头数：{int(results['Transformer']['num_heads'])}")
    lines.append("")
    lines.append("Transformer 多头数对比：")
    for item in head_comparison:
        lines.append(
            f"- heads={int(item['num_heads'])}: 参数量={int(item['params'])}, "
            f"最佳验证损失={item['best_val_loss']:.4f}, 训练时间={item['train_seconds']:.2f}s"
        )
    lines.append("")
    lines.append("长依赖测试摘要：")
    for prompt_name, outputs in dependency_results.items():
        lines.append(f"- {prompt_name}:")
        for model_name, text in outputs.items():
            lines.append(f"  {model_name}: {text}")

    lines.append("")
    lines.append("Checkpoint：")
    for model_name, info in results.items():
        lines.append(f"- {model_name}: {info['checkpoint_path']}")

    save_path.write_text("\n".join(lines), encoding="utf-8")


def save_attention_heatmap(
    model: TinyTransformerLM,
    attention_text: str,
    stoi: dict[str, int],
    save_path: Path,
    device: torch.device,
) -> None:
    """把 Transformer 的平均注意力权重保存为热力图。"""

    model.eval()
    token_ids = torch.tensor([[stoi[ch] for ch in attention_text]], dtype=torch.long, device=device)
    weights = model.get_attention_weights(token_ids)[0].mean(dim=0).cpu()

    seq_len = weights.size(0)
    cell = 18
    margin_left = 90
    margin_top = 90
    width = margin_left + seq_len * cell + 30
    height = margin_top + seq_len * cell + 30
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((20, 20), "Transformer Attention Heatmap", fill="black")
    draw.text((20, 40), f"text: {attention_text}", fill="black")

    for i, ch in enumerate(attention_text):
        label = "_" if ch == " " else ch
        draw.text((margin_left + i * cell + 4, margin_top - 18), label, fill="black")
        draw.text((margin_left - 18, margin_top + i * cell + 4), label, fill="black")

    for row in range(seq_len):
        for col in range(seq_len):
            value = float(weights[row, col])
            shade = 255 - int(220 * value)
            color = (shade, shade, 255)
            x0 = margin_left + col * cell
            y0 = margin_top + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=(210, 210, 210))

    image.save(save_path)


def save_transformer_head_comparison_plot(head_results: list[dict[str, float]], save_path: Path) -> None:
    """保存不同头数 Transformer 的对比图。"""

    width, height = 860, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 85, 70, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    best_vals = [item["best_val_loss"] for item in head_results]
    min_loss = min(best_vals)
    max_loss = max(best_vals)
    padding = max((max_loss - min_loss) * 0.25, 0.0008)
    y_min = min_loss - padding
    y_max = max_loss + padding
    loss_span = max(y_max - y_min, 1e-6)

    draw.rectangle([left, top, left + plot_width, top + plot_height], outline="black", width=2)
    draw.text((left, 18), "Transformer Head Comparison (Zoomed Best Val Loss)", fill="black")
    draw.text((left + plot_width // 2 - 25, height - 25), "num_heads", fill="black")
    draw.text((15, top + plot_height // 2), "best val loss", fill="black")

    y_tick_count = 6
    for i in range(y_tick_count):
        y = top + int(plot_height * i / (y_tick_count - 1))
        draw.line([(left, y), (left + plot_width, y)], fill=(220, 220, 220), width=1)
        tick_value = y_max - (y_max - y_min) * i / (y_tick_count - 1)
        draw.text((22, y - 8), f"{tick_value:.4f}", fill="black")

    x_positions: list[tuple[int, int]] = []
    points: list[tuple[int, int]] = []
    for idx, item in enumerate(head_results):
        x = left + int(plot_width * idx / max(len(head_results) - 1, 1))
        y_ratio = (item["best_val_loss"] - y_min) / loss_span
        y = top + plot_height - int(plot_height * y_ratio)
        x_positions.append((x, int(item["num_heads"])))
        points.append((x, y))

    draw.line(points, fill=(46, 134, 193), width=3)
    for (x, head_num), (_, y), item in zip(x_positions, points, head_results):
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(46, 134, 193), outline="black")
        draw.text((x - 8, top + plot_height + 10), str(head_num), fill="black")
        draw.text((x - 18, y - 20), f"{item['best_val_loss']:.4f}", fill="black")

    image.save(save_path)


def save_attention_heatmaps(
    model: TinyTransformerLM,
    attention_text: str,
    stoi: dict[str, int],
    avg_save_path: Path,
    head_prefix: str,
    device: torch.device,
) -> list[Path]:
    """保存平均注意力热力图和最后一层的分头热力图。"""

    model.eval()
    token_ids = torch.tensor([[stoi[ch] for ch in attention_text]], dtype=torch.long, device=device)
    layer_weights = model.get_attention_weights(token_ids)
    last_layer = layer_weights[-1][0].cpu()
    avg_weights = last_layer.mean(dim=0)

    def draw_heatmap(weights: torch.Tensor, save_path: Path, title: str) -> None:
        seq_len = weights.size(0)
        cell = 18
        margin_left = 90
        margin_top = 90
        width = margin_left + seq_len * cell + 30
        height = margin_top + seq_len * cell + 30
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        draw.text((20, 20), title, fill="black")
        draw.text((20, 40), f"text: {attention_text}", fill="black")
        for i, ch in enumerate(attention_text):
            label = "_" if ch == " " else ch
            draw.text((margin_left + i * cell + 4, margin_top - 18), label, fill="black")
            draw.text((margin_left - 18, margin_top + i * cell + 4), label, fill="black")

        for row in range(seq_len):
            for col in range(seq_len):
                value = float(weights[row, col])
                shade = 255 - int(220 * value)
                color = (shade, shade, 255)
                x0 = margin_left + col * cell
                y0 = margin_top + row * cell
                x1 = x0 + cell
                y1 = y0 + cell
                draw.rectangle((x0, y0, x1, y1), fill=color, outline=(210, 210, 210))

        image.save(save_path)

    draw_heatmap(avg_weights, avg_save_path, "Transformer Attention Heatmap (Average Heads)")
    head_paths: list[Path] = []
    for head_idx in range(last_layer.size(0)):
        head_path = artifact_path(f"{head_prefix}_head{head_idx + 1}.png")
        draw_heatmap(last_layer[head_idx], head_path, f"Transformer Attention Heatmap (Head {head_idx + 1})")
        head_paths.append(head_path)
    return head_paths


def run_transformer_head_comparison(
    vocab_size: int,
    block_size: int,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """对比不同头数的多层 Transformer。"""

    compare_cfg = TrainConfig(
        epochs=25,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        grad_clip=cfg.grad_clip,
        val_ratio=cfg.val_ratio,
    )
    results: list[dict[str, float]] = []
    for num_heads in [1, 2, 4]:
        model = TinyTransformerLM(
            vocab_size=vocab_size,
            block_size=block_size,
            embed_dim=32,
            num_heads=num_heads,
            ffn_dim=96,
            num_layers=2,
        )
        checkpoint_path = artifact_path(f"class2_transformer_heads_{num_heads}.pt")
        metrics = train_language_model(
            model,
            x_data,
            y_data,
            compare_cfg,
            device,
            tag=f"Transformer-{num_heads}H",
            checkpoint_path=checkpoint_path,
            model_config={
                "embed_dim": 32,
                "num_heads": num_heads,
                "ffn_dim": 96,
                "block_size": block_size,
                "num_layers": 2,
            },
        )
        results.append(
            {
                "num_heads": float(num_heads),
                "params": float(count_parameters(model)),
                "best_val_loss": float(metrics["best_val_loss"]),
                "train_seconds": float(metrics["train_seconds"]),
            }
        )
    return results


def inspect_qkv_shapes(
    embed_dim: int = 16,
    seq_len: int = 6,
    batch_size: int = 2,
    num_heads: int = 4,
) -> None:
    """演示多头注意力权重的张量形状。"""

    sample = torch.randn(batch_size, seq_len, embed_dim)
    attn = MultiHeadSelfAttention(embed_dim, num_heads)
    _, weights = attn(sample, causal=True)
    print("Q/K/V 维度演示：")
    print(f"- 输入张量形状: {tuple(sample.shape)}")
    print(f"- 注意力权重形状: {tuple(weights.shape)}")
    print("- 含义: [batch, head, query位置, key位置]")


def main() -> None:
    """程序入口。"""

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"当前使用设备：{device}")
    describe_sequence_modeling_tasks()
    describe_sequence_design_principles()
    demonstrate_gradient_flow()

    corpus = (
        "sequence modeling uses context to predict the next symbol. "
        "rnn keeps a hidden state over time. "
        "lstm uses gates to preserve long term information. "
        "gru also uses gates with fewer parameters. "
        "self attention uses query key value and positional embedding. "
        "transformer replaces recurrence with attention. "
        "the clouds are in the sky. "
        "i grew up in france and now i speak fluent french. "
    )
    corpus = corpus * 12

    stoi, itos = build_char_vocab(corpus)
    token_ids = encode(corpus, stoi)
    block_size = 48
    x_data, y_data = make_lm_dataset(token_ids, block_size)
    cfg = TrainConfig()
    seed_text = "sequence "
    short_prompt = "the clouds are in the "
    long_prompt = "i grew up in france and now i speak fluent "

    histories: dict[str, dict[str, list[float]]] = {}
    generated_samples: dict[str, str] = {}
    dependency_results = {
        "短依赖测试": {},
        "长依赖测试": {},
    }
    comparison_results: dict[str, dict[str, float | str]] = {}
    trained_models: dict[str, nn.Module] = {}

    model_specs = [
        (
            "RNN",
            SimpleRNNLM(vocab_size=len(stoi), embed_dim=24, hidden_dim=64),
            {"embed_dim": 24, "hidden_dim": 64},
        ),
        (
            "LSTM",
            LSTMLM(vocab_size=len(stoi), embed_dim=24, hidden_dim=64),
            {"embed_dim": 24, "hidden_dim": 64},
        ),
        (
            "GRU",
            GRULM(vocab_size=len(stoi), embed_dim=24, hidden_dim=64),
            {"embed_dim": 24, "hidden_dim": 64},
        ),
        (
            "Transformer",
            TinyTransformerLM(
                vocab_size=len(stoi),
                block_size=block_size,
                embed_dim=32,
                num_heads=4,
                ffn_dim=96,
                num_layers=2,
            ),
            {"embed_dim": 32, "num_heads": 4, "ffn_dim": 96, "block_size": block_size, "num_layers": 2},
        ),
    ]

    for tag, model, model_config in model_specs:
        params = count_parameters(model)
        checkpoint_path = artifact_path(f"class2_{tag.lower()}_checkpoint.pt")
        print(f"{tag} 参数量：{params}")
        metrics = train_language_model(
            model,
            x_data,
            y_data,
            cfg,
            device,
            tag=tag,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
        )
        sample_text = generate_text(model, seed_text, stoi, itos, block_size)
        short_text = generate_text(model, short_prompt, stoi, itos, block_size, new_tokens=12)
        long_text = generate_text(model, long_prompt, stoi, itos, block_size, new_tokens=12)

        histories[tag] = {
            "train": metrics["train_losses"],  # type: ignore[assignment]
            "val": metrics["val_losses"],  # type: ignore[assignment]
        }
        generated_samples[tag] = sample_text
        dependency_results["短依赖测试"][tag] = short_text
        dependency_results["长依赖测试"][tag] = long_text
        comparison_results[tag] = {
            "params": float(params),
            "train_seconds": float(metrics["train_seconds"]),
            "avg_epoch_seconds": float(metrics["avg_epoch_seconds"]),
            "final_train_loss": float(metrics["final_train_loss"]),
            "final_val_loss": float(metrics["final_val_loss"]),
            "best_val_loss": float(metrics["best_val_loss"]),
            "sample": sample_text,
            "checkpoint_path": str(checkpoint_path),
            "num_heads": float(model_config.get("num_heads", 0)),
            "num_layers": float(model_config.get("num_layers", 0)),
        }
        trained_models[tag] = model

        print(f"\n{tag} 生成示例：")
        print(sample_text)
        print(f"{tag} 训练耗时：{float(metrics['train_seconds']):.2f}s")
        print(f"{tag} 短依赖测试：{short_text}")
        print(f"{tag} 长依赖测试：{long_text}")

    head_comparison = run_transformer_head_comparison(len(stoi), block_size, x_data, y_data, cfg, device)

    loss_curve_path = artifact_path("class2_training_loss_curves.png")
    generation_report_path = artifact_path("class2_generation_report.txt")
    comparison_report_path = artifact_path("class2_model_comparison.txt")
    attention_heatmap_path = artifact_path("class2_transformer_attention_heatmap.png")
    head_compare_plot_path = artifact_path("class2_transformer_head_comparison.png")

    save_loss_curves(histories, loss_curve_path)
    save_generation_report(generated_samples, dependency_results, generation_report_path)
    save_model_comparison(comparison_results, dependency_results, head_comparison, comparison_report_path)
    save_transformer_head_comparison_plot(head_comparison, head_compare_plot_path)
    head_heatmaps = save_attention_heatmaps(
        trained_models["Transformer"],  # type: ignore[arg-type]
        "i grew up in france and now i speak fluent "[:block_size],
        stoi,
        attention_heatmap_path,
        "class2_transformer_attention",
        device,
    )

    print(f"训练/验证损失曲线已保存到：{loss_curve_path}")
    print(f"生成文本报告已保存到：{generation_report_path}")
    print(f"模型对比报告已保存到：{comparison_report_path}")
    print(f"Transformer 注意力热力图已保存到：{attention_heatmap_path}")

    print(f"Transformer 头数对比图已保存到：{head_compare_plot_path}")
    print(f"Transformer 平均注意力热力图已保存到：{attention_heatmap_path}")
    for path in head_heatmaps:
        print(f"Transformer 分头注意力热力图已保存到：{path}")

    inspect_qkv_shapes()


def run_with_utf8_log() -> None:
    """以 UTF-8 编码同步输出终端日志和文件日志。"""

    log_path = artifact_path("class2_run.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee = Utf8Tee(original_stdout, log_path)
    sys.stdout = tee
    sys.stderr = tee
    try:
        main()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee.close()


if __name__ == "__main__":
    run_with_utf8_log()
