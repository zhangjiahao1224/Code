import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    # 固定随机种子，保证实验可复现
    random.seed(seed)
    torch.manual_seed(seed)


def describe_sequence_modeling_tasks() -> None:
    task_map = {
        "One-to-One": "二分类",
        "Many-to-One": "情感分类",
        "One-to-Many": "图像描述",
        "Many-to-Many": "机器翻译",
    }
    print("序列建模应用场景：")
    for mapping, task in task_map.items():
        print(f"- {mapping}: {task}")


def build_char_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(token_ids: torch.Tensor, itos: dict[int, str]) -> str:
    return "".join(itos[int(i)] for i in token_ids)


def make_lm_dataset(token_ids: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 输入是长度为 block_size 的片段，标签是向右平移一位后的目标序列
    x_list, y_list = [], []
    for i in range(len(token_ids) - block_size):
        x_list.append(token_ids[i : i + block_size])
        y_list.append(token_ids[i + 1 : i + block_size + 1])
    return torch.stack(x_list), torch.stack(y_list)


class SimpleRNNLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        hidden_states, _ = self.rnn(x)
        logits = self.head(hidden_states)
        return logits


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor, causal: bool = True):
        # 由输入向量线性映射得到 Query/Key/Value
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1) / self.scale
        if causal:
            # 因果掩码：禁止当前位置看到未来 token
            t = x.size(1)
            mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attended = attn_weights @ v
        return attended, attn_weights


class TinyTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.attn = SelfAttention(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, causal=True)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.block = TinyTransformerBlock(embed_dim, ffn_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t = x.shape
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(bsz, t)
        token_vec = self.token_embedding(x)
        pos_vec = self.pos_embedding(positions)
        h = token_vec + pos_vec
        h = self.block(h)
        return self.head(h)


@dataclass
class TrainConfig:
    epochs: int = 120
    lr: float = 3e-3
    batch_size: int = 32


def train_language_model(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
    tag: str,
) -> None:
    # 统一训练流程：前向、损失、反向传播、参数更新
    model.to(device)
    dataset = TensorDataset(x_data, y_data)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch == 1 or epoch % 30 == 0:
            avg_loss = running_loss / len(loader)
            print(f"[{tag}] 轮次 {epoch:3d} | 损失={avg_loss:.4f}")


@torch.no_grad()
def generate_text(
    model: nn.Module,
    seed_text: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    block_size: int,
    new_tokens: int = 80,
) -> str:
    # 自回归采样：每次根据当前上下文预测下一个 token
    model.eval()
    device = next(model.parameters()).device
    context = torch.tensor([[stoi[ch] for ch in seed_text]], dtype=torch.long, device=device)

    for _ in range(new_tokens):
        x_cond = context[:, -block_size:]
        logits = model(x_cond)
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)

    return decode(context[0].cpu(), itos)


def inspect_qkv_shapes(embed_dim: int = 16, seq_len: int = 6, batch_size: int = 2) -> None:
    sample = torch.randn(batch_size, seq_len, embed_dim)
    attn = SelfAttention(embed_dim)
    _, weights = attn(sample, causal=True)
    print("Q/K/V 维度演示：")
    print(f"- 输入张量形状: {tuple(sample.shape)}")
    print(f"- 注意力权重形状: {tuple(weights.shape)}")
    print("- 含义: [batch, query位置, key位置]")


def main() -> None:
    set_seed(42)
    describe_sequence_modeling_tasks()

    # 构造一个简短重复语料，用于快速演示语言建模训练
    corpus = (
        "sequence modeling uses context to predict the next symbol. "
        "rnn keeps a hidden state over time. "
        "self attention uses query key value and positional embedding. "
        "transformer replaces recurrence with attention.\n"
    )
    corpus = corpus * 20

    stoi, itos = build_char_vocab(corpus)
    token_ids = encode(corpus, stoi)

    block_size = 32
    x_data, y_data = make_lm_dataset(token_ids, block_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TrainConfig(epochs=120, lr=3e-3, batch_size=64)

    rnn_model = SimpleRNNLM(vocab_size=len(stoi), embed_dim=24, hidden_dim=64)
    train_language_model(rnn_model, x_data, y_data, cfg, device, tag="RNN")
    rnn_text = generate_text(rnn_model, "sequence ", stoi, itos, block_size)
    print("\nRNN 生成示例：")
    print(rnn_text)

    transformer_model = TinyTransformerLM(
        vocab_size=len(stoi),
        block_size=block_size,
        embed_dim=32,
        ffn_dim=96,
    )
    train_language_model(transformer_model, x_data, y_data, cfg, device, tag="Transformer")
    tfm_text = generate_text(transformer_model, "sequence ", stoi, itos, block_size)
    print("\nTransformer 生成示例：")
    print(tfm_text)

    inspect_qkv_shapes()


if __name__ == "__main__":
    main()
