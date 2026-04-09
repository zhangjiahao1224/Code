import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# 本文件演示字符级语言建模（character-level language modeling）：
# 1. 先把一段英文语料拆成单个字符，并建立“字符 <-> 数字编号”的映射。
# 2. 再把连续文本切成训练样本，让模型根据前面的字符预测下一个字符。
# 3. 最后分别训练一个 RNN 语言模型和一个简化版 Transformer 语言模型，并生成文本。


def set_seed(seed: int = 42) -> None:
    """固定随机种子，让每次运行时的数据顺序和模型初始化尽量一致。"""

    # 固定随机种子，保证实验可复现
    random.seed(seed)
    torch.manual_seed(seed)


def describe_sequence_modeling_tasks() -> None:
    """打印常见的序列建模任务类型，帮助理解输入序列和输出序列的对应关系。"""

    # key 表示输入/输出序列的长度关系，value 给出一个典型应用例子。
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
    """根据语料构建字符级词表。

    stoi: string to index，把字符转换成整数编号，方便送入神经网络。
    itos: index to string，把整数编号转换回字符，方便把模型输出还原成文本。
    """

    # set(text) 去重，sorted(...) 固定顺序，保证同样语料得到同样的编号。
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """把字符串编码成 token id 张量。

    这里的 token 是“字符”，所以每个字符都会被替换为它在词表中的整数编号。
    """

    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(token_ids: torch.Tensor, itos: dict[int, str]) -> str:
    """把 token id 张量解码回字符串。"""

    return "".join(itos[int(i)] for i in token_ids)


def make_lm_dataset(token_ids: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """构造语言模型训练数据。

    x 是长度为 block_size 的上下文片段。
    y 是 x 向右平移一位后的目标序列，也就是每个位置应该预测的“下一个字符”。
    例如文本编号为 [a, b, c, d] 且 block_size=3，则 x=[a, b, c]，y=[b, c, d]。
    """

    # 输入是长度为 block_size 的片段，标签是向右平移一位后的目标序列。
    x_list, y_list = [], []
    for i in range(len(token_ids) - block_size):
        # 第 i 个样本使用 token_ids[i : i + block_size] 作为模型输入。
        x_list.append(token_ids[i : i + block_size])
        # 目标比输入整体晚一个位置，用来训练“预测下一个 token”。
        y_list.append(token_ids[i + 1 : i + block_size + 1])

    # torch.stack 会把多个一维序列堆叠成二维张量：[样本数, block_size]。
    return torch.stack(x_list), torch.stack(y_list)


class SimpleRNNLM(nn.Module):
    """一个简单的 RNN 字符级语言模型。

    模型结构：
    token id -> Embedding 向量 -> RNN 隐状态 -> Linear 输出每个字符的预测分数。
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        # Embedding 把离散的字符编号映射成连续向量，形状从 [B, T] 变为 [B, T, embed_dim]。
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # batch_first=True 表示输入/输出张量使用 [batch, time, feature] 的维度顺序。
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # head 把每个时间步的 hidden_dim 隐状态映射到 vocab_size 个字符的 logits。
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，返回每个位置对下一个字符的预测分数 logits。"""

        x = self.embedding(x)
        # hidden_states 包含序列中每个位置的隐藏状态；这里不需要最后的隐藏状态，所以用 _ 忽略。
        hidden_states, _ = self.rnn(x)
        # logits 形状为 [batch, time, vocab_size]，后续会和 y 计算交叉熵损失。
        logits = self.head(hidden_states)
        return logits


class SelfAttention(nn.Module):
    """单头自注意力层。

    Self-Attention 的核心思想：序列中每个位置都可以根据 Query 和 Key 的相似度，
    从所有允许看到的位置的 Value 中聚合信息。
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # Query 用来表示“当前位置想找什么信息”。
        self.query = nn.Linear(embed_dim, embed_dim)
        # Key 用来表示“每个位置能提供什么索引信息”。
        self.key = nn.Linear(embed_dim, embed_dim)
        # Value 用来表示“每个位置真正要被汇总的内容”。
        self.value = nn.Linear(embed_dim, embed_dim)
        # 缩放因子可以避免点积结果过大，减少 softmax 饱和。
        self.scale = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor, causal: bool = True):
        """计算自注意力输出。

        x 的形状通常是 [batch, time, embed_dim]。
        causal=True 时使用因果掩码，保证当前位置不能看到未来 token，适合语言模型。
        """

        # 由输入向量线性映射得到 Query/Key/Value。
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # q @ k^T 得到每个 query 位置对每个 key 位置的注意力打分。
        # transpose(-2, -1) 交换 time 和 embed_dim 维度，便于做矩阵乘法。
        scores = q @ k.transpose(-2, -1) / self.scale
        if causal:
            # 因果掩码：禁止当前位置看到未来 token。
            # 上三角区域代表“未来位置”，会被填成 -inf，让 softmax 后权重接近 0。
            t = x.size(1)
            mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        # softmax 把注意力分数归一化为概率分布，每个 query 位置的权重和为 1。
        attn_weights = torch.softmax(scores, dim=-1)
        # 用注意力权重对 Value 加权求和，得到每个位置融合上下文后的表示。
        attended = attn_weights @ v
        return attended, attn_weights


class TinyTransformerBlock(nn.Module):
    """一个极简 Transformer Block。

    这里包含：单头自注意力、残差连接、LayerNorm 和前馈网络。
    为了教学清晰，省略了多头注意力、Dropout 等更完整模型中常见的组件。
    """

    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.attn = SelfAttention(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        # 前馈网络逐位置处理每个 token 表示：先升维到 ffn_dim，再降回 embed_dim。
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行一次 Transformer Block 前向传播。"""

        attn_out, _ = self.attn(x, causal=True)
        # 残差连接 x + attn_out 可以缓解深层网络训练困难；LayerNorm 稳定数值分布。
        x = self.ln1(x + attn_out)
        # 前馈网络进一步变换每个位置的表示，再做一次残差连接和归一化。
        x = self.ln2(x + self.ffn(x))
        return x


class TinyTransformerLM(nn.Module):
    """一个简化版 Transformer 字符级语言模型。"""

    def __init__(self, vocab_size: int, block_size: int, embed_dim: int, ffn_dim: int):
        super().__init__()
        # block_size 表示模型一次最多看多少个历史 token。
        self.block_size = block_size
        # token_embedding 学习“每个字符本身的语义表示”。
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # pos_embedding 学习“字符在序列中第几个位置”的位置信息。
        # Transformer 本身没有循环结构，所以必须显式加入位置编码。
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.block = TinyTransformerBlock(embed_dim, ffn_dim)
        # 把 Transformer 输出的隐藏表示映射成词表大小的预测分数。
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回每个时间步对下一个字符的预测分数。"""

        bsz, t = x.shape
        # positions 形状为 [batch, time]，每一行都是 0 到 t-1 的位置编号。
        positions = torch.arange(t, device=x.device).unsqueeze(0).expand(bsz, t)
        token_vec = self.token_embedding(x)
        pos_vec = self.pos_embedding(positions)
        # token 向量和位置向量相加，让模型同时知道“是什么字符”和“在什么位置”。
        h = token_vec + pos_vec
        h = self.block(h)
        return self.head(h)


@dataclass
class TrainConfig:
    """训练超参数配置。"""

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
    """统一训练流程：前向传播、计算损失、反向传播、参数更新。"""

    # 把模型移动到 CPU 或 GPU，取决于 device 的值。
    model.to(device)
    # TensorDataset 把输入 x_data 和标签 y_data 绑定起来，保证每次取样时一一对应。
    dataset = TensorDataset(x_data, y_data)
    # DataLoader 负责按 batch 取数据；shuffle=True 可以打乱样本顺序，提升训练稳定性。
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    # Adam 是常用优化器，会根据梯度自动调整每个参数的更新幅度。
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        # train() 开启训练模式。这里没有 Dropout/BatchNorm，但保留是良好习惯。
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            # cross_entropy 期望输入形状为 [N, C]，标签为 [N]。
            # 因此把 [batch, time, vocab] 展平成 [batch*time, vocab]。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

            # 标准训练三步：清空旧梯度 -> 反向传播计算新梯度 -> 优化器更新参数。
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
    """使用训练好的语言模型生成文本。

    自回归采样：每次根据当前上下文预测下一个 token，
    再把预测出的 token 拼回上下文，继续预测后面的 token。
    """

    # eval() 切换到评估模式；@torch.no_grad() 禁止梯度计算，加快推理并节省显存。
    model.eval()
    device = next(model.parameters()).device
    # seed_text 是生成的起始文本，先转换成 token id，形状为 [1, 当前长度]。
    context = torch.tensor([[stoi[ch] for ch in seed_text]], dtype=torch.long, device=device)

    for _ in range(new_tokens):
        # 只保留最近 block_size 个 token，避免超过 Transformer 的位置编码长度。
        x_cond = context[:, -block_size:]
        logits = model(x_cond)
        # 语言模型只需要最后一个位置的输出，因为它代表“下一个 token”的预测分布。
        next_logits = logits[:, -1, :]
        probs = torch.softmax(next_logits, dim=-1)
        # multinomial 按概率采样，比直接 argmax 更有随机性，生成文本更丰富。
        next_token = torch.multinomial(probs, num_samples=1)
        # 把新采样出来的 token 拼接到上下文末尾，进入下一轮预测。
        context = torch.cat([context, next_token], dim=1)

    return decode(context[0].cpu(), itos)


def inspect_qkv_shapes(embed_dim: int = 16, seq_len: int = 6, batch_size: int = 2) -> None:
    """用随机输入演示自注意力中注意力权重的张量形状。"""

    # sample 模拟一批序列表示，形状为 [batch_size, seq_len, embed_dim]。
    sample = torch.randn(batch_size, seq_len, embed_dim)
    attn = SelfAttention(embed_dim)
    _, weights = attn(sample, causal=True)
    print("Q/K/V 维度演示：")
    print(f"- 输入张量形状: {tuple(sample.shape)}")
    print(f"- 注意力权重形状: {tuple(weights.shape)}")
    print("- 含义: [batch, query位置, key位置]")


def main() -> None:
    """程序入口：准备数据、训练两个模型、生成示例文本并打印注意力维度。"""

    set_seed(42)
    describe_sequence_modeling_tasks()

    # 构造一个简短重复语料，用于快速演示语言建模训练
    corpus = (
        "sequence modeling uses context to predict the next symbol. "
        "rnn keeps a hidden state over time. "
        "self attention uses query key value and positional embedding. "
        "transformer replaces recurrence with attention.\n"
    )
    # 重复语料可以增加训练样本数量，让小模型更容易在短时间内学到模式。
    corpus = corpus * 20

    # 构建字符词表并把整段语料编码成整数序列。
    stoi, itos = build_char_vocab(corpus)
    token_ids = encode(corpus, stoi)

    # block_size 是模型每次看到的上下文窗口长度。
    block_size = 32
    x_data, y_data = make_lm_dataset(token_ids, block_size)
    # 如果机器支持 CUDA 就使用 GPU，否则使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 统一训练配置：训练轮数、学习率、batch 大小。
    cfg = TrainConfig(epochs=120, lr=3e-3, batch_size=64)

    # 先训练 RNN 语言模型，并用同一个 seed_text 生成文本，方便和 Transformer 对比。
    rnn_model = SimpleRNNLM(vocab_size=len(stoi), embed_dim=24, hidden_dim=64)
    train_language_model(rnn_model, x_data, y_data, cfg, device, tag="RNN")
    rnn_text = generate_text(rnn_model, "sequence ", stoi, itos, block_size)
    print("\nRNN 生成示例：")
    print(rnn_text)

    # 再训练简化版 Transformer 语言模型。
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

    # 最后打印 Q/K/V 注意力权重的维度，帮助理解 Self-Attention 的张量形状。
    inspect_qkv_shapes()


if __name__ == "__main__":
    main()
