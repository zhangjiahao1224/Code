from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ------------------------------------------------------------
# 第 9 课：从零拆解一个教学型 Transformer Encoder
#
# 这份脚本的目标不是追求大模型规模，而是把 Transformer 的关键结构
# 拆成清晰、可运行、可观察的代码：
#
# 1. 构造一个合成序列任务：
#    输入是一串 token，其中包含一个特殊标记 <MARK>。
#    模型需要预测 <MARK> 后面的那个数字。
#
# 2. 从底层模块搭建 Transformer：
#    - Token Embedding：把离散 token id 映射为连续向量。
#    - Position Embedding：给模型注入顺序信息。
#    - Scaled Dot-Product Attention：计算 query 与 key 的相似度。
#    - Multi-Head Self-Attention：让模型从多个子空间观察同一段序列。
#    - Feed Forward Network：对每个位置的表示做非线性变换。
#    - Residual + LayerNorm：稳定深层网络训练。
#
# 3. 训练一个小型 Transformer Encoder 分类器。
#
# 4. 导出 artifacts：
#    - 模型 checkpoint
#    - 训练曲线
#    - 注意力热力图
#    - 架构与预测报告
#
# 这个任务非常适合用来理解 Transformer 为什么比普通 MLP 更适合序列：
# 输出标签不是由固定位置决定，而是由“<MARK> 在哪里”动态决定。
# 模型必须先定位 <MARK>，再读取它后面的 token。
# ------------------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "Class_9"


def artifact_path(filename: str) -> Path:
    """返回第 9 课统一的产物路径，并确保目录存在。"""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / filename


def set_seed(seed: int = 42) -> None:
    """固定随机种子，让每次运行的训练曲线和示例尽量可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """统计模型中所有可训练参数的数量。"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass(frozen=True)
class TrainConfig:
    """集中管理第 9 课实验超参数。

    把这些值放进 dataclass 的好处是：后续想比较层数、头数、序列长度、
    answer offset 或训练轮数时，不需要在 main() 里到处找魔法数字。
    """

    seed: int = 42
    train_content_len: int = 10
    max_content_len: int = 14
    answer_offset: int = 1
    train_samples: int = 2400
    val_samples: int = 600
    batch_size: int = 64
    embed_dim: int = 64
    num_heads: int = 4
    ffn_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    epochs: int = 18
    lr: float = 3e-4
    patience: int = 4
    min_delta: float = 1e-4
    target_val_acc: float = 0.999
    min_epochs_before_target_stop: int = 6
    comparison_epochs: int = 10
    comparison_patience: int = 2
    decoder_content_len: int = 10
    decoder_train_samples: int = 1800
    decoder_val_samples: int = 400
    decoder_epochs: int = 10
    decoder_lr: float = 4e-4
    seq2seq_content_len: int = 6
    seq2seq_train_samples: int = 2600
    seq2seq_val_samples: int = 500
    seq2seq_epochs: int = 12
    seq2seq_lr: float = 4e-4


@dataclass(frozen=True)
class TrainResult:
    """训练循环的结构化返回值。"""

    history: dict[str, list[float]]
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_val_loss: float
    best_val_acc: float
    stopped_early: bool


@dataclass(frozen=True)
class Vocabulary:
    """本实验使用的迷你词表。

    token 设计：
    - <CLS>：放在序列最前面，用来展示常见 Transformer 输入格式。
    - <MARK>：提示模型“答案在我后面一格”。
    - 0~9：普通数字 token，也是分类标签的 10 个类别。
    """

    cls_id: int = 0
    mark_id: int = 1
    digit_offset: int = 2
    num_digits: int = 10

    @property
    def vocab_size(self) -> int:
        return self.digit_offset + self.num_digits

    def digit_to_id(self, digit: int) -> int:
        return self.digit_offset + digit

    def id_to_text(self, token_id: int) -> str:
        if token_id == self.cls_id:
            return "<CLS>"
        if token_id == self.mark_id:
            return "<MARK>"
        digit = token_id - self.digit_offset
        if 0 <= digit < self.num_digits:
            return str(digit)
        return f"<UNK:{token_id}>"

    def decode(self, token_ids: torch.Tensor | list[int]) -> list[str]:
        """把 token id 序列解码成人类可读 token 列表。

        注意：训练时模型使用的是整数 token id；decode 只用于打印报告、
        标注 attention heatmap 和帮助我们检查样本内容。
        """

        return [self.id_to_text(int(token_id)) for token_id in token_ids]

    def decode_to_text(self, token_ids: torch.Tensor | list[int]) -> str:
        """把 token id 序列解码成一个空格分隔的字符串。"""

        return " ".join(self.decode(token_ids))


class MarkerLookupDataset(Dataset):
    """带标记符的合成序列分类数据集。

    每个样本都遵循同一个规则：
    - 第 0 位永远是 <CLS>。
    - 后面是一串数字 token。
    - 其中某个位置会被替换为 <MARK>。
    - 标签是 <MARK> 正后方的那个数字。

    举例：
    输入：<CLS> 8 1 <MARK> 7 2 9
    标签：7

    这个任务故意让 <MARK> 的位置随机变化。
    如果模型只记固定位置，就无法稳定解决问题；它必须学会使用注意力
    根据上下文动态查找答案位置。

    训练时我们会从 <MARK> 位置的最终隐藏状态做分类。
    这类似在序列里放了一个“查询 token”：它的问题是“我后面是谁？”
    因此注意力热力图里最值得看的就是 <MARK> 那一行是否关注答案 token。
    """

    def __init__(
        self,
        num_samples: int,
        content_len: int,
        vocab: Vocabulary,
        seed: int,
        answer_offset: int = 1,
    ):
        super().__init__()
        if answer_offset < 1:
            raise ValueError("answer_offset 至少为 1。")
        if content_len <= answer_offset:
            raise ValueError("content_len 必须大于 answer_offset，才能放下 <MARK> 和答案 token。")

        generator = random.Random(seed)
        self.vocab = vocab
        self.answer_offset = answer_offset
        self.samples: list[tuple[torch.Tensor, torch.Tensor, int]] = []

        for _ in range(num_samples):
            digits = [generator.randrange(vocab.num_digits) for _ in range(content_len)]

            # marker_index 是 content 区域内的位置。
            # 它不能太靠后，否则就没有 marker 后第 answer_offset 个 token 作为答案。
            marker_index = generator.randrange(0, content_len - answer_offset)
            answer_digit = digits[marker_index + answer_offset]

            token_ids = [vocab.cls_id]
            for idx, digit in enumerate(digits):
                if idx == marker_index:
                    token_ids.append(vocab.mark_id)
                else:
                    token_ids.append(vocab.digit_to_id(digit))

            x = torch.tensor(token_ids, dtype=torch.long)
            y = torch.tensor(answer_digit, dtype=torch.long)
            # 保存 marker 在完整序列里的位置，便于后面做可解释性报告。
            marker_position = marker_index + 1
            self.samples.append((x, y, marker_position))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, marker_position = self.samples[index]
        return x, y, torch.tensor(marker_position, dtype=torch.long)


class CountingLanguageModelDataset(Dataset):
    """用于 Decoder-only Transformer 的自回归语言建模数据集。

    每个样本是一段从随机数字开始、按 mod 10 递增的序列：
    <CLS> 3 4 5 6 7 ...

    训练目标是 next-token prediction：
    输入：<CLS> 3 4 5
    标签：3     4 5 6

    这类任务必须使用 causal mask。否则模型在预测当前位置标签时可以直接看见
    右侧未来 token，就不再是“自回归生成”了。
    """

    def __init__(self, num_samples: int, content_len: int, vocab: Vocabulary, seed: int):
        super().__init__()
        if content_len < 2:
            raise ValueError("content_len 至少为 2，才能形成 next-token 训练样本。")

        generator = random.Random(seed)
        self.vocab = vocab
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for _ in range(num_samples):
            start = generator.randrange(vocab.num_digits)
            digits = [(start + step) % vocab.num_digits for step in range(content_len)]
            full_sequence = [vocab.cls_id] + [vocab.digit_to_id(digit) for digit in digits]
            x = torch.tensor(full_sequence[:-1], dtype=torch.long)
            y = torch.tensor(full_sequence[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


class Seq2SeqCopyDataset(Dataset):
    """用于完整 Encoder-Decoder Transformer 的序列到序列 copy 任务。

    source 是一串随机数字 token，例如：
    source:        8 1 4 4 9 2

    Decoder 采用 teacher forcing：
    decoder_input: <CLS> 8 1 4 4 9
    target:        8     1 4 4 9 2

    因为 source 是随机的，Decoder 不能只靠左侧历史预测下一个随机数字。
    它必须通过 cross-attention 去读取 Encoder 产生的 source memory。
    """

    def __init__(self, num_samples: int, content_len: int, vocab: Vocabulary, seed: int):
        super().__init__()
        if content_len < 2:
            raise ValueError("content_len 至少为 2。")

        generator = random.Random(seed)
        self.vocab = vocab
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for _ in range(num_samples):
            digits = [generator.randrange(vocab.num_digits) for _ in range(content_len)]
            source_ids = [vocab.digit_to_id(digit) for digit in digits]
            decoder_input_ids = [vocab.cls_id] + source_ids[:-1]
            target_ids = source_ids
            self.samples.append(
                (
                    torch.tensor(source_ids, dtype=torch.long),
                    torch.tensor(decoder_input_ids, dtype=torch.long),
                    torch.tensor(target_ids, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[index]


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention 的最小实现。

    输入形状约定：
    - query: [batch, heads, seq_len, head_dim]
    - key:   [batch, heads, seq_len, head_dim]
    - value: [batch, heads, seq_len, head_dim]

    计算步骤：
    1. query 与 key 做点积，得到每个位置对其他位置的关注分数。
    2. 除以 sqrt(head_dim)，避免维度变大后点积数值过大。
    3. softmax，把分数变成概率分布。
    4. 用注意力权重对 value 加权求和，得到新的上下文表示。
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        # mask 中为 False 的位置会被填成极小值，softmax 后近似为 0。
        # 本实验没有 padding，但保留 mask 参数可以展示真实工程里的常见接口。
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, value)
        return context, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力模块。

    Self-Attention 的意思是 query、key、value 都来自同一个输入序列。
    Multi-Head 的意思是把 embedding 维度切成多个头，每个头独立计算注意力。

    这样做的直觉是：
    - 一个头可以学会定位 <MARK>。
    - 另一个头可以学会关注 <MARK> 后面的答案 token。
    - 还有的头可能关注局部邻居或全局 <CLS>。
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除。")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 一次线性层同时生成 Q、K、V，比写三个 Linear 更紧凑。
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把 [B, T, C] 改成 [B, H, T, D]，方便逐头计算注意力。"""

        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """把 [B, H, T, D] 合回 [B, T, C]，交给输出投影层。"""

        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(x)
        query, key, value = qkv.chunk(3, dim=-1)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        context, attention_weights = self.attention(query, key, value, mask=mask)
        context = self._merge_heads(context)
        output = self.out_proj(context)
        output = self.dropout(output)
        return output, attention_weights


class MultiHeadCrossAttention(nn.Module):
    """Encoder-Decoder Transformer 中的多头交叉注意力。

    Self-Attention 是 Q/K/V 都来自同一个序列。
    Cross-Attention 则不同：
    - query 来自 Decoder 当前隐藏状态。
    - key/value 来自 Encoder 输出的 source memory。

    它回答的问题是：Decoder 当前要生成某个位置时，应该读取源序列里的哪些位置？
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除。")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim)
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.embed_dim)

    def forward(
        self,
        query_states: torch.Tensor,
        memory_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._split_heads(self.q_proj(query_states))
        key, value = self.kv_proj(memory_states).chunk(2, dim=-1)
        key = self._split_heads(key)
        value = self._split_heads(value)

        context, attention_weights = self.attention(query, key, value, mask=mask)
        output = self.out_proj(self._merge_heads(context))
        output = self.dropout(output)
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """Transformer Block 中逐位置使用的前馈网络。

    注意：这里的 FFN 不会在序列位置之间通信。
    序列位置之间的信息交换已经由 Self-Attention 完成。
    FFN 的作用是对每个位置当前的表示做更强的非线性变换。
    """

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """一个 Pre-LN Transformer Encoder Block。

    本实现采用 Pre-LN 结构：
    - 先 LayerNorm，再进入 Attention / FFN。
    - 输出再通过残差连接加回原输入。

    与 Post-LN 相比，Pre-LN 在较深网络里通常更稳定。
    对教学脚本来说，它也能让小模型更容易训练。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_attn = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.ln_attn(x)
        attn_output, attention_weights = self.self_attn(attn_input, mask=mask)
        x = x + attn_output

        ffn_input = self.ln_ffn(x)
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output
        return x, attention_weights


class TinyTransformerClassifier(nn.Module):
    """用于标记查找任务的小型 Transformer Encoder 分类器。"""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        num_classes: int,
        readout_token_id: int,
        readout_mode: str = "mark",
        use_position_embedding: bool = True,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if readout_mode not in {"mark", "cls"}:
            raise ValueError("readout_mode 只能是 'mark' 或 'cls'。")

        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.readout_token_id = readout_token_id
        self.readout_mode = readout_mode
        self.use_position_embedding = use_position_embedding

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim) if use_position_embedding else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        token_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"输入长度 {seq_len} 超过模型最大长度 {self.max_seq_len}。")

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)

        x = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            x = x + self.position_embedding(positions)
        x = self.dropout(x)

        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            x, attention_weights = block(x)
            attention_maps.append(attention_weights)

        x = self.final_norm(x)

        if self.readout_mode == "cls":
            # 对比实验会用这个分支：只读取 <CLS>。它必须先把答案信息汇聚回第 0 位，
            # 因此比直接读取 <MARK> 位置更难。
            readout_state = x[:, 0, :]
        else:
            # 本任务的答案由 <MARK> 决定，所以分类头读取 <MARK> 位置的最终表示。
            # 真实项目里也常见这种模式：先用一个特殊 token 表示查询意图，
            # 再让它通过 Self-Attention 从整段序列里取回需要的信息。
            readout_mask = token_ids == self.readout_token_id
            if not torch.all(readout_mask.any(dim=1)):
                raise ValueError("每个输入样本都必须包含 readout_token_id。")
            readout_positions = readout_mask.float().argmax(dim=1)
            batch_indices = torch.arange(batch_size, device=token_ids.device)
            readout_state = x[batch_indices, readout_positions, :]
        logits = self.classifier(readout_state)

        if return_attention:
            return logits, attention_maps
        return logits


class TransformerDecoderBlock(nn.Module):
    """一个 Decoder-only Transformer Block。

    它和 Encoder Block 的主要差别不是层的数量，而是 self-attention
    必须带 causal mask：第 t 个位置只能看 0..t，不能看 t 右边的未来 token。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_attn = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_input = self.ln_attn(x)
        attn_output, attention_weights = self.self_attn(attn_input, mask=causal_mask)
        x = x + attn_output
        x = x + self.ffn(self.ln_ffn(x))
        return x, attention_weights


class TinyTransformerDecoderLM(nn.Module):
    """Decoder-only Transformer 语言模型。

    这个模型不做分类，而是对每个位置都预测“下一个 token 是什么”。
    它的输出形状是 [batch, seq_len, vocab_size]。
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self._cached_mask: torch.Tensor | None = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """返回 [1, 1, T, T] 的下三角 mask，True 表示允许关注。

        内部缓存 mask，只在 seq_len 增长或设备变更时重建。
        """

        if self._cached_mask is not None and self._cached_mask.size(-1) >= seq_len:
            cached = self._cached_mask
            if cached.device != device:
                cached = cached.to(device)
                self._cached_mask = cached
            return cached[:, :, :seq_len, :seq_len]

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        self._cached_mask = mask.view(1, 1, seq_len, seq_len)
        return self._cached_mask

    def forward(
        self,
        token_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"输入长度 {seq_len} 超过 Decoder 最大长度 {self.max_seq_len}。")

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = self._causal_mask(seq_len, token_ids.device)
        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            x, attention_weights = block(x, causal_mask=causal_mask)
            attention_maps.append(attention_weights)

        logits = self.lm_head(self.final_norm(x))
        if return_attention:
            return logits, attention_maps
        return logits


class TransformerSeq2SeqDecoderBlock(nn.Module):
    """完整 Transformer Decoder Block：masked self-attention + cross-attention + FFN。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_self_attn = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ln_cross_attn = nn.LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout=dropout)
        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_attn_input = self.ln_self_attn(x)
        self_attn_output, self_attention = self.self_attn(self_attn_input, mask=causal_mask)
        x = x + self_attn_output

        cross_attn_input = self.ln_cross_attn(x)
        cross_attn_output, cross_attention = self.cross_attn(cross_attn_input, memory)
        x = x + cross_attn_output

        x = x + self.ffn(self.ln_ffn(x))
        return x, self_attention, cross_attention


class TinyTransformerSeq2Seq(nn.Module):
    """完整 Encoder-Decoder Transformer。

    Encoder 负责把 source 序列编码成 memory。
    Decoder 先用 causal self-attention 读取已生成前缀，再用 cross-attention
    从 Encoder memory 中检索源序列信息。
    """

    def __init__(
        self,
        vocab_size: int,
        max_source_len: int,
        max_target_len: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.source_embedding = nn.Embedding(vocab_size, embed_dim)
        self.source_position = nn.Embedding(max_source_len, embed_dim)
        self.target_embedding = nn.Embedding(vocab_size, embed_dim)
        self.target_position = nn.Embedding(max_target_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, ffn_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerSeq2SeqDecoderBlock(embed_dim, num_heads, ffn_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self._cached_mask: torch.Tensor | None = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """返回 [1, 1, T, T] 的下三角 mask，内部缓存避免重复创建。"""

        if self._cached_mask is not None and self._cached_mask.size(-1) >= seq_len:
            cached = self._cached_mask
            if cached.device != device:
                cached = cached.to(device)
                self._cached_mask = cached
            return cached[:, :, :seq_len, :seq_len]

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        self._cached_mask = mask.view(1, 1, seq_len, seq_len)
        return self._cached_mask

    def encode(self, source_ids: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size, source_len = source_ids.shape
        if source_len > self.max_source_len:
            raise ValueError(f"source 长度 {source_len} 超过最大长度 {self.max_source_len}。")

        positions = torch.arange(source_len, device=source_ids.device).unsqueeze(0).expand(batch_size, source_len)
        x = self.source_embedding(source_ids) + self.source_position(positions)
        x = self.dropout(x)
        encoder_attention: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            x, attention = block(x)
            encoder_attention.append(attention)
        return self.encoder_norm(x), encoder_attention

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batch_size, target_len = decoder_input_ids.shape
        if target_len > self.max_target_len:
            raise ValueError(f"target 长度 {target_len} 超过最大长度 {self.max_target_len}。")

        positions = torch.arange(target_len, device=decoder_input_ids.device).unsqueeze(0).expand(batch_size, target_len)
        x = self.target_embedding(decoder_input_ids) + self.target_position(positions)
        x = self.dropout(x)
        causal_mask = self._causal_mask(target_len, decoder_input_ids.device)

        self_attention_maps: list[torch.Tensor] = []
        cross_attention_maps: list[torch.Tensor] = []
        for block in self.decoder_blocks:
            x, self_attention, cross_attention = block(x, memory, causal_mask)
            self_attention_maps.append(self_attention)
            cross_attention_maps.append(cross_attention)

        logits = self.lm_head(self.decoder_norm(x))
        return logits, self_attention_maps, cross_attention_maps

    def forward(
        self,
        source_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        memory, encoder_attention = self.encode(source_ids)
        logits, decoder_self_attention, cross_attention = self.decode(decoder_input_ids, memory)
        if return_attention:
            return logits, encoder_attention, decoder_self_attention, cross_attention
        return logits


def _make_loaders(
    vocab: Vocabulary,
    dataset_type: str,
    content_len: int,
    train_samples: int,
    val_samples: int,
    batch_size: int,
    train_seed: int,
    val_seed: int,
    answer_offset: int = 1,
) -> tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """统一的 DataLoader 工厂，根据 dataset_type 创建不同数据集。

    dataset_type 可选："marker" / "decoder" / "seq2seq"。
    """

    if dataset_type == "marker":
        train_ds = MarkerLookupDataset(
            train_samples, content_len, vocab=vocab, seed=train_seed, answer_offset=answer_offset
        )
        val_ds = MarkerLookupDataset(
            val_samples, content_len, vocab=vocab, seed=val_seed, answer_offset=answer_offset
        )
    elif dataset_type == "decoder":
        train_ds = CountingLanguageModelDataset(train_samples, content_len, vocab=vocab, seed=train_seed)
        val_ds = CountingLanguageModelDataset(val_samples, content_len, vocab=vocab, seed=val_seed)
    elif dataset_type == "seq2seq":
        train_ds = Seq2SeqCopyDataset(train_samples, content_len, vocab=vocab, seed=train_seed)
        val_ds = Seq2SeqCopyDataset(val_samples, content_len, vocab=vocab, seed=val_seed)
    else:
        raise ValueError(f"未知的 dataset_type：{dataset_type}，可选 'marker' / 'decoder' / 'seq2seq'。")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_ds, val_ds


def _process_batch(
    model: nn.Module,
    batch,
    device: torch.device,
    mode: str,
) -> tuple[torch.Tensor, int, int]:
    """统一的前向传播 + loss 计算，返回 (loss, num_correct, num_tokens)。

    mode 参数区分三种模型的前向接口：
    - "classifier"：读取 (token_ids, labels, _)，返回 logits → 分类 loss。
    - "decoder"：跳过第 0 个 token（prompt），做 token-level 评估。
    - "seq2seq"：接收 (source, decoder_input, target) 三元组。
    """

    if mode == "classifier":
        token_ids, labels, _ = batch
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        logits = model(token_ids)
        loss = F.cross_entropy(logits, labels)
        correct = int((logits.argmax(dim=1) == labels).sum().item())
        return loss, correct, token_ids.size(0)

    if mode == "decoder":
        token_ids, labels = batch
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        logits = model(token_ids)
        # 第 0 个目标是随机起点，只凭 <CLS> 无法确定；这里把它当作 prompt，不计入指标。
        scored_logits = logits[:, 1:, :]
        scored_labels = labels[:, 1:]
        loss = F.cross_entropy(
            scored_logits.reshape(-1, scored_logits.size(-1)),
            scored_labels.reshape(-1),
        )
        correct = int((scored_logits.argmax(dim=-1) == scored_labels).sum().item())
        return loss, correct, scored_labels.numel()

    if mode == "seq2seq":
        source_ids, decoder_input_ids, target_ids = batch
        source_ids = source_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        target_ids = target_ids.to(device)
        logits = model(source_ids, decoder_input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )
        correct = int((logits.argmax(dim=-1) == target_ids).sum().item())
        return loss, correct, target_ids.numel()

    raise ValueError(f"未知的 mode：{mode}，可选 'classifier' / 'decoder' / 'seq2seq'。")


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mode: str = "classifier",
) -> tuple[float, float]:
    """统一的评估：返回 (平均 loss, accuracy)。"""

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            loss, correct, count = _process_batch(model, batch, device, mode)
            total_loss += float(loss.item()) * count
            total_correct += correct
            total_count += count

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def build_model(
    vocab: Vocabulary,
    config: TrainConfig,
    readout_mode: str = "mark",
    use_position_embedding: bool = True,
    num_heads: int | None = None,
    num_layers: int | None = None,
    answer_offset: int | None = None,
) -> TinyTransformerClassifier:
    """按配置创建 Transformer 分类器，便于主实验和对比实验复用。"""

    heads = config.num_heads if num_heads is None else num_heads
    layers = config.num_layers if num_layers is None else num_layers
    offset = config.answer_offset if answer_offset is None else answer_offset
    # max_seq_len 留出更长序列压力测试所需的位置 embedding。
    max_seq_len = config.max_content_len + 1
    if config.max_content_len <= offset:
        raise ValueError("max_content_len 必须大于 answer_offset。")

    return TinyTransformerClassifier(
        vocab_size=vocab.vocab_size,
        max_seq_len=max_seq_len,
        num_classes=vocab.num_digits,
        readout_token_id=vocab.mark_id,
        readout_mode=readout_mode,
        use_position_embedding=use_position_embedding,
        embed_dim=config.embed_dim,
        num_heads=heads,
        ffn_dim=config.ffn_dim,
        num_layers=layers,
        dropout=config.dropout,
    )


def build_decoder_model(vocab: Vocabulary, config: TrainConfig) -> TinyTransformerDecoderLM:
    """按第 9 课配置创建一个 Decoder-only 语言模型。"""

    max_seq_len = config.decoder_content_len + 8
    return TinyTransformerDecoderLM(
        vocab_size=vocab.vocab_size,
        max_seq_len=max_seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def build_seq2seq_model(vocab: Vocabulary, config: TrainConfig) -> TinyTransformerSeq2Seq:
    """按第 9 课配置创建完整 Encoder-Decoder Transformer。"""

    return TinyTransformerSeq2Seq(
        vocab_size=vocab.vocab_size,
        max_source_len=config.seq2seq_content_len,
        max_target_len=config.seq2seq_content_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    mode: str,
    epochs: int = 18,
    lr: float = 3e-4,
    patience: int = 4,
    min_delta: float = 1e-4,
    target_val_acc: float = 0.999,
    min_epochs_before_target_stop: int = 6,
    tag: str = "main",
) -> TrainResult:
    """统一的训练循环，通过 mode 参数适配三种模型。

    mode：
    - "classifier"：Encoder 分类器，batch = (token_ids, labels, marker_pos)。
    - "decoder"：Decoder-only LM，batch = (token_ids, labels)。
    - "seq2seq"：Encoder-Decoder，batch = (source, decoder_input, target)。

    所有模式共享同一套 early stopping、梯度裁剪和日志逻辑。
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in train_loader:
            loss, correct, count = _process_batch(model, batch, device, mode)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * count
            total_correct += correct
            total_count += count

        train_loss = total_loss / max(total_count, 1)
        train_acc = total_correct / max(total_count, 1)
        val_loss, val_acc = _evaluate(model, val_loader, device, mode)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[{tag}] epoch={epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if epoch >= min_epochs_before_target_stop and val_acc >= target_val_acc:
            stopped_early = True
            print(
                f"[{tag}] target accuracy reached: "
                f"epoch={epoch}, val_acc={val_acc:.3f}, target={target_val_acc:.3f}"
            )
            break

        if epochs_without_improvement >= patience:
            stopped_early = True
            print(f"[{tag}] early stopping: best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
            break

    return TrainResult(
        history=history,
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        stopped_early=stopped_early,
    )


def save_checkpoint(
    model: TinyTransformerClassifier,
    result: TrainResult,
    path: Path,
) -> Path:
    """保存模型权重、结构参数和训练历史。"""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "history": result.history,
        "config": {
            "max_seq_len": model.max_seq_len,
            "embed_dim": model.embed_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "readout_token_id": model.readout_token_id,
            "readout_mode": model.readout_mode,
            "use_position_embedding": model.use_position_embedding,
        },
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "best_val_acc": result.best_val_acc,
        "stopped_early": result.stopped_early,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def _load_font(size: int = 15):
    """优先加载支持中文的字体；失败时退回 Pillow 默认字体。"""

    try:
        from PIL import ImageFont
    except ImportError:
        return None

    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _safe_text(draw, xy: tuple[int, int], text: str, fill: str, font) -> None:
    """字体不支持中文时，自动降级为 ASCII，避免画图阶段报错。"""

    try:
        draw.text(xy, text, fill=fill, font=font)
    except UnicodeEncodeError:
        fallback = text.encode("ascii", errors="ignore").decode("ascii") or "text"
        draw.text(xy, fallback, fill=fill, font=font)


def _text_size(draw, text: str, font) -> tuple[int, int]:
    """返回文字宽高，用于在图表单元格中居中摆放标签。"""

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except UnicodeEncodeError:
        fallback = text.encode("ascii", errors="ignore").decode("ascii") or "text"
        bbox = draw.textbbox((0, 0), fallback, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _safe_centered_text(
    draw,
    box: tuple[int, int, int, int],
    text: str,
    fill: str,
    font,
) -> None:
    """在指定矩形内居中绘制文字。"""

    text_width, text_height = _text_size(draw, text, font)
    x0, y0, x1, y1 = box
    x = x0 + max((x1 - x0 - text_width) // 2, 0)
    y = y0 + max((y1 - y0 - text_height) // 2, 0)
    _safe_text(draw, (x, y), text, fill, font)


def save_training_curves(history: dict[str, list[float]], path: Path) -> Path | None:
    """把 loss 和 accuracy 曲线保存为 PNG 图片。"""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("未安装 Pillow，跳过训练曲线图片导出。")
        return None

    width, height = 980, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(15)
    title_font = _load_font(20)

    _safe_text(draw, (40, 20), "Class 9 Transformer Training Curves", "black", title_font)

    panels = [
        ("Loss", {"train_loss": history["train_loss"], "val_loss": history["val_loss"]}, 70, 80),
        ("Accuracy", {"train_acc": history["train_acc"], "val_acc": history["val_acc"]}, 70, 350),
    ]
    colors = {
        "train_loss": (33, 150, 243),
        "val_loss": (244, 67, 54),
        "train_acc": (0, 150, 136),
        "val_acc": (255, 152, 0),
    }

    for title, series_map, left, top in panels:
        plot_width = 840
        plot_height = 210
        right = left + plot_width
        bottom = top + plot_height
        draw.rectangle([left, top, right, bottom], outline="black", width=2)
        _safe_text(draw, (left, top - 28), title, "black", font)

        values = [value for series in series_map.values() for value in series]
        vmin = min(values)
        vmax = max(values)
        if "Accuracy" in title:
            vmin, vmax = 0.0, 1.02
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1.0

        for tick_idx in range(5):
            ratio = tick_idx / 4
            y = bottom - int(ratio * plot_height)
            value = vmin + ratio * (vmax - vmin)
            draw.line([(left, y), (right, y)], fill=(235, 235, 235), width=1)
            _safe_text(draw, (18, y - 8), f"{value:.2f}", "gray", font)

        max_len = max(len(series) for series in series_map.values())

        def project(idx: int, value: float) -> tuple[int, int]:
            x = left + int(idx / max(max_len - 1, 1) * plot_width)
            y = bottom - int((value - vmin) / max(vmax - vmin, 1e-8) * plot_height)
            return x, y

        legend_x = right - 190
        legend_y = top + 14
        for row_idx, (name, series) in enumerate(series_map.items()):
            points = [project(idx, value) for idx, value in enumerate(series)]
            color = colors[name]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for x, y in points:
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
            draw.rectangle(
                [legend_x, legend_y + row_idx * 22, legend_x + 16, legend_y + 12 + row_idx * 22],
                fill=color,
            )
            _safe_text(draw, (legend_x + 24, legend_y - 4 + row_idx * 22), name, "black", font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_comparison_plot(results: list[dict[str, float | str]], path: Path) -> Path | None:
    """保存若干 Transformer 变体的验证集准确率对比图。"""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("未安装 Pillow，跳过对比实验图片导出。")
        return None

    if not results:
        return None

    width, height = 980, 480
    left, top, right, bottom = 90, 90, 940, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(14)
    title_font = _load_font(20)

    _safe_text(draw, (42, 26), "Class 9 Transformer Ablation Study", "black", title_font)
    _safe_text(draw, (42, 54), "validation accuracy after short training", "gray", font)
    draw.rectangle([left, top, right, bottom], outline="black", width=2)

    for tick_idx in range(5):
        ratio = tick_idx / 4
        y = bottom - int(ratio * (bottom - top))
        value = ratio
        draw.line([(left, y), (right, y)], fill=(235, 235, 235), width=1)
        _safe_text(draw, (28, y - 8), f"{value:.2f}", "gray", font)

    bar_gap = 20
    bar_width = max(36, int((right - left - bar_gap * (len(results) + 1)) / len(results)))
    colors = [(25, 118, 210), (0, 137, 123), (251, 140, 0), (142, 36, 170), (198, 40, 40), (85, 139, 47)]

    for idx, item in enumerate(results):
        acc = float(item["best_val_acc"])
        name = str(item["name"])
        x0 = left + bar_gap + idx * (bar_width + bar_gap)
        x1 = x0 + bar_width
        y0 = bottom - int(acc * (bottom - top))
        color = colors[idx % len(colors)]
        draw.rectangle([x0, y0, x1, bottom], fill=color)
        _safe_centered_text(draw, (x0 - 10, y0 - 26, x1 + 10, y0 - 4), f"{acc:.2f}", "black", font)
        _safe_centered_text(draw, (x0 - 20, bottom + 12, x1 + 20, bottom + 58), name, "black", font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_attention_heatmap(
    attention: torch.Tensor,
    tokens: list[str],
    marker_position: int,
    answer_position: int,
    path: Path,
    title: str = "Class 9 Transformer Attention Heatmap",
) -> Path | None:
    """保存最后一层平均多头注意力热力图。

    attention 形状：[heads, seq_len, seq_len]。
    横轴表示“被关注的位置”，纵轴表示“发出 query 的位置”。
    """

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("未安装 Pillow，跳过注意力热力图导出。")
        return None

    avg_attention = attention.detach().cpu().mean(dim=0)
    seq_len = avg_attention.size(0)
    cell = 64
    left = 132
    top = 112
    width = left + seq_len * cell + 34
    height = top + seq_len * cell + 72

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(14)
    title_font = _load_font(18)

    _safe_text(draw, (24, 20), title, "black", title_font)
    _safe_text(
        draw,
        (24, 48),
        f"marker={marker_position}, answer={answer_position}; darker means higher attention",
        "gray",
        font,
    )

    for idx, token in enumerate(tokens):
        x0 = left + idx * cell
        y0 = top + idx * cell
        _safe_centered_text(draw, (x0, top - 34, x0 + cell, top - 10), token, "black", font)
        _safe_centered_text(draw, (12, y0, left - 12, y0 + cell), token, "black", font)

    for row in range(seq_len):
        for col in range(seq_len):
            value = float(avg_attention[row, col].item())
            intensity = int(255 - value * 230)
            intensity = max(20, min(255, intensity))
            color = (255, intensity, intensity)

            x0 = left + col * cell
            y0 = top + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(230, 230, 230))

            text_color = "black" if intensity > 120 else "white"
            _safe_centered_text(draw, (x0, y0, x1, y1), f"{value:.2f}", text_color, font)

    # 用边框标出 <MARK> 和答案列，方便阅读热力图。
    for col, outline in [(marker_position, (25, 118, 210)), (answer_position, (0, 137, 123))]:
        x0 = left + col * cell
        x1 = x0 + cell
        draw.rectangle([x0, top, x1, top + seq_len * cell], outline=outline, width=4)

    # 再额外标出 <MARK> 行，因为这一行表示“查询 token 正在看哪里”。
    marker_y0 = top + marker_position * cell
    marker_y1 = marker_y0 + cell
    draw.rectangle([left, marker_y0, left + seq_len * cell, marker_y1], outline=(25, 118, 210), width=3)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_marker_attention_bar(
    attention: torch.Tensor,
    tokens: list[str],
    marker_position: int,
    answer_position: int,
    path: Path,
) -> Path | None:
    """把 <MARK> 行的注意力单独画成条形图，方便观察答案 token 是否被重点关注。"""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("未安装 Pillow，跳过 <MARK> attention 条形图导出。")
        return None

    values = attention.detach().cpu().mean(dim=0)[marker_position]
    width, height = 980, 420
    left, top, right, bottom = 88, 90, 940, 315
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(14)
    title_font = _load_font(20)

    _safe_text(draw, (42, 26), "Attention From <MARK> Token", "black", title_font)
    _safe_text(draw, (42, 54), f"green bar is answer position {answer_position}", "gray", font)
    draw.rectangle([left, top, right, bottom], outline="black", width=2)

    max_value = max(float(values.max().item()), 1e-6)
    bar_gap = 10
    bar_width = max(28, int((right - left - bar_gap * (len(tokens) + 1)) / len(tokens)))

    for idx, token in enumerate(tokens):
        value = float(values[idx].item())
        x0 = left + bar_gap + idx * (bar_width + bar_gap)
        x1 = x0 + bar_width
        y0 = bottom - int(value / max_value * (bottom - top))
        color = (0, 137, 123) if idx == answer_position else (25, 118, 210)
        if idx == marker_position:
            color = (251, 140, 0)
        draw.rectangle([x0, y0, x1, bottom], fill=color)
        _safe_centered_text(draw, (x0 - 8, y0 - 24, x1 + 8, y0 - 4), f"{value:.2f}", "black", font)
        _safe_centered_text(draw, (x0 - 14, bottom + 10, x1 + 14, bottom + 42), token, "black", font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_attention_diagnostics(
    attention_maps: list[torch.Tensor],
    tokens: list[str],
    marker_position: int,
    answer_position: int,
) -> list[Path]:
    """导出平均热力图、逐层热力图、最后一层分头热力图和 <MARK> 条形图。"""

    saved_paths: list[Path] = []
    if not attention_maps:
        return saved_paths

    last_layer = attention_maps[-1]
    avg_path = save_attention_heatmap(
        last_layer,
        tokens,
        marker_position,
        answer_position,
        artifact_path("class9_transformer_attention_heatmap.png"),
        title="Class 9 Attention Heatmap (Last Layer Average)",
    )
    if avg_path is not None:
        saved_paths.append(avg_path)

    for layer_idx, layer_attention in enumerate(attention_maps, start=1):
        layer_path = save_attention_heatmap(
            layer_attention,
            tokens,
            marker_position,
            answer_position,
            artifact_path(f"class9_transformer_attention_layer{layer_idx}.png"),
            title=f"Class 9 Attention Heatmap (Layer {layer_idx} Average)",
        )
        if layer_path is not None:
            saved_paths.append(layer_path)

    for head_idx in range(last_layer.size(0)):
        head_path = save_attention_heatmap(
            last_layer[head_idx : head_idx + 1],
            tokens,
            marker_position,
            answer_position,
            artifact_path(f"class9_transformer_attention_last_layer_head{head_idx + 1}.png"),
            title=f"Class 9 Attention Heatmap (Last Layer Head {head_idx + 1})",
        )
        if head_path is not None:
            saved_paths.append(head_path)

    bar_path = save_marker_attention_bar(
        last_layer,
        tokens,
        marker_position,
        answer_position,
        artifact_path("class9_transformer_marker_attention_bar.png"),
    )
    if bar_path is not None:
        saved_paths.append(bar_path)

    return saved_paths


def save_decoder_attention_heatmap(
    attention: torch.Tensor,
    tokens: list[str],
    path: Path,
) -> Path | None:
    """保存 Decoder causal self-attention 热力图。"""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("未安装 Pillow，跳过 Decoder 注意力热力图导出。")
        return None

    avg_attention = attention.detach().cpu().mean(dim=0)
    seq_len = avg_attention.size(0)
    cell = 64
    left = 132
    top = 112
    width = left + seq_len * cell + 34
    height = top + seq_len * cell + 72

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(14)
    title_font = _load_font(18)
    _safe_text(draw, (24, 20), "Class 9 Decoder Causal Attention Heatmap", "black", title_font)
    _safe_text(draw, (24, 48), "upper-right future positions should stay near zero", "gray", font)

    for idx, token in enumerate(tokens):
        x0 = left + idx * cell
        y0 = top + idx * cell
        _safe_centered_text(draw, (x0, top - 34, x0 + cell, top - 10), token, "black", font)
        _safe_centered_text(draw, (12, y0, left - 12, y0 + cell), token, "black", font)

    for row in range(seq_len):
        for col in range(seq_len):
            value = float(avg_attention[row, col].item())
            intensity = max(20, min(255, int(255 - value * 230)))
            color = (255, intensity, intensity)
            if col > row:
                color = (238, 238, 238)
            x0 = left + col * cell
            y0 = top + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(225, 225, 225))
            text_color = "black" if intensity > 120 or col > row else "white"
            _safe_centered_text(draw, (x0, y0, x1, y1), f"{value:.2f}", text_color, font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_cross_attention_heatmap(
    attention: torch.Tensor,
    source_tokens: list[str],
    target_tokens: list[str],
    path: Path,
) -> Path | None:
    """保存 Encoder-Decoder cross-attention 热力图。

    横轴是 Encoder source token，纵轴是 Decoder target position。
    """

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("未安装 Pillow，跳过 cross-attention 热力图导出。")
        return None

    avg_attention = attention.detach().cpu().mean(dim=0)
    target_len, source_len = avg_attention.shape
    cell = 64
    left = 132
    top = 112
    width = left + source_len * cell + 34
    height = top + target_len * cell + 72

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(14)
    title_font = _load_font(18)
    _safe_text(draw, (24, 20), "Class 9 Encoder-Decoder Cross-Attention", "black", title_font)
    _safe_text(draw, (24, 48), "rows are decoder positions, columns are encoder source tokens", "gray", font)

    for idx, token in enumerate(source_tokens):
        x0 = left + idx * cell
        _safe_centered_text(draw, (x0, top - 34, x0 + cell, top - 10), token, "black", font)
    for idx, token in enumerate(target_tokens):
        y0 = top + idx * cell
        _safe_centered_text(draw, (12, y0, left - 12, y0 + cell), token, "black", font)

    for row in range(target_len):
        for col in range(source_len):
            value = float(avg_attention[row, col].item())
            intensity = max(20, min(255, int(255 - value * 230)))
            color = (255, intensity, intensity)
            x0 = left + col * cell
            y0 = top + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(225, 225, 225))
            text_color = "black" if intensity > 120 else "white"
            _safe_centered_text(draw, (x0, y0, x1, y1), f"{value:.2f}", text_color, font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def collect_prediction_examples(
    model: TinyTransformerClassifier,
    dataset: MarkerLookupDataset,
    vocab: Vocabulary,
    device: torch.device,
    num_examples: int = 8,
) -> tuple[list[str], list[torch.Tensor], list[str], int, int]:
    """收集若干预测示例，并返回第一条样本的注意力用于画图。"""

    model.eval()
    lines: list[str] = []
    first_attention_maps: list[torch.Tensor] | None = None
    first_tokens: list[str] = []
    first_marker_position = 0
    first_answer_position = 0

    with torch.no_grad():
        for idx in range(num_examples):
            token_ids, label, marker_position = dataset[idx]
            batch = token_ids.unsqueeze(0).to(device)
            logits, attention_maps = model(batch, return_attention=True)
            pred = int(logits.argmax(dim=1).item())
            prob = float(F.softmax(logits, dim=1)[0, pred].item())

            tokens = vocab.decode(token_ids.tolist())
            answer_position = int(marker_position.item()) + dataset.answer_offset
            marker_text = tokens[int(marker_position.item())]
            answer_text = tokens[answer_position]

            lines.append(
                " | ".join(
                    [
                        f"sample={idx + 1}",
                        f"input={' '.join(tokens)}",
                        f"marker={marker_text}@{int(marker_position.item())}",
                        f"answer_token={answer_text}@{answer_position}",
                        f"label={int(label.item())}",
                        f"pred={pred}",
                        f"confidence={prob:.3f}",
                    ]
                )
            )

            if first_attention_maps is None:
                first_attention_maps = [layer_attention[0].detach().cpu() for layer_attention in attention_maps]
                first_tokens = tokens
                first_marker_position = int(marker_position.item())
                first_answer_position = answer_position

    if first_attention_maps is None:
        raise RuntimeError("没有收集到预测示例。")
    return lines, first_attention_maps, first_tokens, first_marker_position, first_answer_position


def save_decode_demo(
    dataset: MarkerLookupDataset,
    vocab: Vocabulary,
    path: Path,
    num_examples: int = 5,
) -> Path:
    """保存 token id -> token text 的 decode 示例。

    Transformer 的训练输入始终是整数 id；decode 的价值是让人能读懂样本、
    报告和注意力图。这个 artifact 把两种视角并排放在一起。
    """

    lines = [
        "Class 9 Token Decode Demo",
        "",
        "说明：",
        "- 模型训练时看到的是 token_ids，也就是整数序列。",
        "- Vocabulary.decode(token_ids) 会把整数转回 <CLS>、<MARK> 和数字 token。",
        "- decode 不参与训练和反向传播，只用于报告、可视化和人工检查。",
        "",
    ]

    for idx in range(min(num_examples, len(dataset))):
        token_ids, label, marker_position = dataset[idx]
        token_id_list = [int(token_id) for token_id in token_ids.tolist()]
        tokens = vocab.decode(token_id_list)
        decoded_text = vocab.decode_to_text(token_id_list)
        marker_idx = int(marker_position.item())
        answer_idx = marker_idx + dataset.answer_offset

        lines.extend(
            [
                f"sample {idx + 1}",
                f"token_ids     : {token_id_list}",
                f"decoded_tokens: {tokens}",
                f"decoded_text  : {decoded_text}",
                f"marker        : token_ids[{marker_idx}]={token_id_list[marker_idx]} -> {tokens[marker_idx]}",
                f"answer        : token_ids[{answer_idx}]={token_id_list[answer_idx]} -> {tokens[answer_idx]}",
                f"label         : {int(label.item())}",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def generate_with_decoder(
    model: TinyTransformerDecoderLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
) -> list[int]:
    """用 Decoder-only 模型按 greedy decoding 生成后续 token。"""

    model.eval()
    generated = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = torch.tensor([generated[-model.max_seq_len :]], dtype=torch.long, device=device)
            logits = model(context)
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            generated.append(next_id)
    return generated


def save_decoder_report(
    model: TinyTransformerDecoderLM,
    result: TrainResult,
    dataset: CountingLanguageModelDataset,
    vocab: Vocabulary,
    device: torch.device,
    path: Path,
) -> tuple[Path, Path | None]:
    """保存 Decoder-only 实验报告，并返回报告和注意力图路径。"""

    model.eval()
    sample_x, sample_y = dataset[0]
    with torch.no_grad():
        logits, attention_maps = model(sample_x.unsqueeze(0).to(device), return_attention=True)
        pred_ids = logits.argmax(dim=-1)[0].cpu()

    prompt = [vocab.cls_id, vocab.digit_to_id(3)]
    generated = generate_with_decoder(model, prompt, max_new_tokens=10, device=device)
    attention_path = save_decoder_attention_heatmap(
        attention_maps[-1][0].detach().cpu(),
        vocab.decode(sample_x.tolist()),
        artifact_path("class9_decoder_causal_attention_heatmap.png"),
    )

    lines = [
        "第 9 课：Transformer Decoder 自回归生成实验报告",
        "",
        "一、Decoder 任务",
        "- 输入是 <CLS> 开头的递增数字序列。",
        "- 训练目标是 next-token prediction：每个位置预测下一个 token。",
        "- 第 0 个目标是随机起点，只凭 <CLS> 无法确定，因此训练和评估时不计入 loss/accuracy。",
        "- Decoder 使用 causal mask，因此当前位置不能关注右侧未来 token。",
        "",
        "二、模型结构",
        f"- 参数量：{count_parameters(model)}",
        f"- 最大序列长度：{model.max_seq_len}",
        f"- embedding 维度：{model.embed_dim}",
        f"- Transformer Decoder 层数：{model.num_layers}",
        f"- attention heads：{model.num_heads}",
        "",
        "三、训练结果",
        f"- final_train_loss：{result.history['train_loss'][-1]:.4f}",
        f"- final_train_token_acc：{result.history['train_acc'][-1]:.4f}",
        f"- final_val_loss：{result.history['val_loss'][-1]:.4f}",
        f"- final_val_token_acc：{result.history['val_acc'][-1]:.4f}",
        f"- best_epoch：{result.best_epoch}",
        f"- best_val_loss：{result.best_val_loss:.4f}",
        f"- best_val_token_acc：{result.best_val_acc:.4f}",
        "",
        "四、样本预测",
        f"input_ids      : {[int(x) for x in sample_x.tolist()]}",
        f"input_text     : {vocab.decode_to_text(sample_x.tolist())}",
        f"target_text    : {vocab.decode_to_text(sample_y.tolist())}",
        f"predicted_text : {vocab.decode_to_text(pred_ids.tolist())}",
        f"scored_target  : {vocab.decode_to_text(sample_y[1:].tolist())}",
        f"scored_pred    : {vocab.decode_to_text(pred_ids[1:].tolist())}",
        "",
        "五、Greedy 生成",
        f"prompt_text    : {vocab.decode_to_text(prompt)}",
        f"generated_text : {vocab.decode_to_text(generated)}",
        "",
        "六、读图提示",
        "- causal attention heatmap 的右上角代表未来位置。",
        "- 右上角应接近 0，说明 Decoder 没有偷看未来 token。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, attention_path


def save_decoder_checkpoint(
    model: TinyTransformerDecoderLM,
    result: TrainResult,
    path: Path,
) -> Path:
    """保存 Decoder-only 模型 checkpoint。"""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "history": result.history,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "best_val_acc": result.best_val_acc,
        "config": {
            "max_seq_len": model.max_seq_len,
            "embed_dim": model.embed_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "vocab_size": model.vocab_size,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def greedy_decode_seq2seq(
    model: TinyTransformerSeq2Seq,
    source_ids: torch.Tensor,
    vocab: Vocabulary,
    max_tokens: int,
    device: torch.device,
) -> list[int]:
    """用完整 Encoder-Decoder Transformer 做 greedy decoding。"""

    model.eval()
    source_batch = source_ids.unsqueeze(0).to(device)
    generated = [vocab.cls_id]
    with torch.no_grad():
        memory, _encoder_attention = model.encode(source_batch)
        for _ in range(max_tokens):
            decoder_input = torch.tensor([generated[-model.max_target_len :]], dtype=torch.long, device=device)
            logits, _self_attention, _cross_attention = model.decode(decoder_input, memory)
            next_id = int(logits[0, -1].argmax(dim=-1).item())
            generated.append(next_id)
    return generated[1:]


def save_seq2seq_checkpoint(
    model: TinyTransformerSeq2Seq,
    result: TrainResult,
    path: Path,
) -> Path:
    """保存完整 Encoder-Decoder Transformer checkpoint。"""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "history": result.history,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "best_val_acc": result.best_val_acc,
        "config": {
            "max_source_len": model.max_source_len,
            "max_target_len": model.max_target_len,
            "embed_dim": model.embed_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "vocab_size": model.vocab_size,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def save_seq2seq_report(
    model: TinyTransformerSeq2Seq,
    result: TrainResult,
    dataset: Seq2SeqCopyDataset,
    vocab: Vocabulary,
    device: torch.device,
    path: Path,
) -> tuple[Path, Path | None]:
    """保存完整 Encoder-Decoder Transformer 报告和 cross-attention 图。"""

    model.eval()
    source_ids, decoder_input_ids, target_ids = dataset[0]
    with torch.no_grad():
        logits, encoder_attention, decoder_self_attention, cross_attention = model(
            source_ids.unsqueeze(0).to(device),
            decoder_input_ids.unsqueeze(0).to(device),
            return_attention=True,
        )
        pred_ids = logits.argmax(dim=-1)[0].cpu()

    generated_ids = greedy_decode_seq2seq(model, source_ids, vocab, max_tokens=target_ids.numel(), device=device)
    cross_attention_path = save_cross_attention_heatmap(
        cross_attention[-1][0].detach().cpu(),
        source_tokens=vocab.decode(source_ids.tolist()),
        target_tokens=vocab.decode(decoder_input_ids.tolist()),
        path=artifact_path("class9_seq2seq_cross_attention_heatmap.png"),
    )

    lines = [
        "第 9 课：完整 Transformer Encoder-Decoder 实验报告",
        "",
        "一、Seq2Seq 任务",
        "- Encoder 输入一串随机数字 source。",
        "- Decoder 从 <CLS> 开始，用 teacher forcing 预测同一串数字。",
        "- Decoder block 包含 masked self-attention、cross-attention 和 FFN。",
        "- cross-attention 的 query 来自 Decoder，key/value 来自 Encoder memory。",
        "",
        "二、模型结构",
        f"- 参数量：{count_parameters(model)}",
        f"- source 最大长度：{model.max_source_len}",
        f"- target 最大长度：{model.max_target_len}",
        f"- embedding 维度：{model.embed_dim}",
        f"- Encoder/Decoder 层数：{model.num_layers}",
        f"- attention heads：{model.num_heads}",
        "",
        "三、训练结果",
        f"- final_train_loss：{result.history['train_loss'][-1]:.4f}",
        f"- final_train_token_acc：{result.history['train_acc'][-1]:.4f}",
        f"- final_val_loss：{result.history['val_loss'][-1]:.4f}",
        f"- final_val_token_acc：{result.history['val_acc'][-1]:.4f}",
        f"- best_epoch：{result.best_epoch}",
        f"- best_val_loss：{result.best_val_loss:.4f}",
        f"- best_val_token_acc：{result.best_val_acc:.4f}",
        "",
        "四、样本预测",
        f"source_text        : {vocab.decode_to_text(source_ids.tolist())}",
        f"decoder_input_text : {vocab.decode_to_text(decoder_input_ids.tolist())}",
        f"target_text        : {vocab.decode_to_text(target_ids.tolist())}",
        f"teacher_pred_text  : {vocab.decode_to_text(pred_ids.tolist())}",
        f"greedy_output_text : {vocab.decode_to_text(generated_ids)}",
        "",
        "五、读图提示",
        "- cross-attention heatmap 的横轴是 Encoder source token。",
        "- 纵轴是 Decoder 当前 token 位置。",
        "- 如果某一行在对应 source token 上权重大，说明 Decoder 正在从 Encoder memory 取数。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, cross_attention_path


def run_comparison_experiments(
    vocab: Vocabulary,
    config: TrainConfig,
    device: torch.device,
) -> list[dict[str, float | str]]:
    """运行轻量级 ablation study，观察 Transformer 组件选择的影响。"""

    variants = [
        {
            "name": "mark-readout",
            "readout_mode": "mark",
            "use_position_embedding": True,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "answer_offset": 1,
        },
        {
            "name": "cls-readout",
            "readout_mode": "cls",
            "use_position_embedding": True,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "answer_offset": 1,
        },
        {
            "name": "no-position",
            "readout_mode": "mark",
            "use_position_embedding": False,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "answer_offset": 1,
        },
        {
            "name": "one-head",
            "readout_mode": "mark",
            "use_position_embedding": True,
            "num_heads": 1,
            "num_layers": config.num_layers,
            "answer_offset": 1,
        },
        {
            "name": "one-layer",
            "readout_mode": "mark",
            "use_position_embedding": True,
            "num_heads": config.num_heads,
            "num_layers": 1,
            "answer_offset": 1,
        },
        {
            "name": "offset-2",
            "readout_mode": "mark",
            "use_position_embedding": True,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "answer_offset": 2,
        },
    ]

    results: list[dict[str, float | str]] = []
    print("\n开始轻量对比实验：")
    for idx, variant in enumerate(variants, start=1):
        set_seed(config.seed + 100 + idx)
        answer_offset = int(variant["answer_offset"])
        train_loader, val_loader, _train_dataset, _val_dataset = _make_loaders(
            vocab=vocab,
            dataset_type="marker",
            content_len=config.train_content_len,
            train_samples=1200,
            val_samples=360,
            batch_size=config.batch_size,
            answer_offset=answer_offset,
            train_seed=config.seed + 200 + idx,
            val_seed=config.seed + 300 + idx,
        )
        model = build_model(
            vocab,
            config,
            readout_mode=str(variant["readout_mode"]),
            use_position_embedding=bool(variant["use_position_embedding"]),
            num_heads=int(variant["num_heads"]),
            num_layers=int(variant["num_layers"]),
            answer_offset=answer_offset,
        ).to(device)

        result = _train_model(
            model,
            train_loader,
            val_loader,
            device,
            mode="classifier",
            epochs=config.comparison_epochs,
            lr=config.lr,
            patience=config.comparison_patience,
            min_delta=config.min_delta,
            target_val_acc=config.target_val_acc,
            min_epochs_before_target_stop=config.min_epochs_before_target_stop,
            tag=str(variant["name"]),
        )
        results.append(
            {
                "name": str(variant["name"]),
                "best_val_loss": result.best_val_loss,
                "best_val_acc": result.best_val_acc,
                "best_epoch": float(result.best_epoch),
                "params": float(count_parameters(model)),
            }
        )

    return results


def evaluate_long_sequence_generalization(
    model: TinyTransformerClassifier,
    vocab: Vocabulary,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """用更长序列做一次压力测试，观察长度变化带来的影响。"""

    long_dataset = MarkerLookupDataset(
        num_samples=600,
        content_len=config.max_content_len,
        vocab=vocab,
        seed=config.seed + 909,
        answer_offset=config.answer_offset,
    )
    long_loader = DataLoader(long_dataset, batch_size=config.batch_size, shuffle=False)
    val_loss, val_acc = _evaluate(model, long_loader, device, mode="classifier")
    return {
        "content_len": float(config.max_content_len),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }


def save_report(
    model: TinyTransformerClassifier,
    result: TrainResult,
    example_lines: list[str],
    comparison_results: list[dict[str, float | str]],
    long_eval: dict[str, float] | None,
    path: Path,
) -> Path:
    """保存结构说明、训练结果和预测样例。"""

    history = result.history
    comparison_lines = [
        f"- {item['name']}：best_val_acc={float(item['best_val_acc']):.4f}, "
        f"best_val_loss={float(item['best_val_loss']):.4f}, params={int(float(item['params']))}"
        for item in comparison_results
    ]
    long_eval_lines = []
    if long_eval is not None:
        long_eval_lines = [
            "",
            "五、长序列压力测试",
            f"- content_len：{int(long_eval['content_len'])}",
            f"- val_loss：{long_eval['val_loss']:.4f}",
            f"- val_acc：{long_eval['val_acc']:.4f}",
            "- 这个测试会使用训练时较少出现或没有充分训练的位置 embedding，因此它更像是在观察长度外推风险。",
        ]

    lines = [
        "第 9 课：Transformer Encoder 标记查找实验报告",
        "",
        "一、任务规则",
        "- 输入序列第 0 位是 <CLS>，用于展示标准 Transformer 输入格式。",
        "- 序列中有一个 <MARK>。",
        "- 标签是 <MARK> 后面一格的数字。",
        "- 分类头读取 <MARK> 位置的最终隐藏状态。",
        "- 因为 <MARK> 位置随机，模型必须通过注意力动态查找答案。",
        "",
        "二、模型结构",
        f"- 参数量：{count_parameters(model)}",
        f"- 最大序列长度：{model.max_seq_len}",
        f"- embedding 维度：{model.embed_dim}",
        f"- Transformer 层数：{model.num_layers}",
        f"- attention heads：{model.num_heads}",
        f"- readout token id：{model.readout_token_id}",
        f"- readout mode：{model.readout_mode}",
        f"- use position embedding：{model.use_position_embedding}",
        "- 每层结构：LayerNorm -> MultiHeadSelfAttention -> Residual -> LayerNorm -> FFN -> Residual",
        "",
        "三、训练结果",
        f"- final_train_loss：{history['train_loss'][-1]:.4f}",
        f"- final_train_acc：{history['train_acc'][-1]:.4f}",
        f"- final_val_loss：{history['val_loss'][-1]:.4f}",
        f"- final_val_acc：{history['val_acc'][-1]:.4f}",
        f"- best_epoch：{result.best_epoch}",
        f"- best_val_loss：{result.best_val_loss:.4f}",
        f"- best_val_acc：{result.best_val_acc:.4f}",
        f"- stopped_early：{result.stopped_early}",
        "",
        "四、对比实验",
        *comparison_lines,
        *long_eval_lines,
        "",
        "六、预测示例",
        *example_lines,
        "",
        "七、读图提示",
        "- attention heatmap 的横轴是被关注 token，纵轴是发出 query 的 token。",
        "- 蓝色边框标出 <MARK> 所在列，绿色边框标出答案 token 所在列。",
        "- 如果 <MARK> 行明显关注答案位置，说明模型已经学会用查询 token 取回目标信息。",
        "- 分头 heatmap 可以观察不同 head 是否学到不同关注模式。",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    """程序入口：训练模型并导出所有第 9 课 artifacts。"""

    config = TrainConfig()
    set_seed(config.seed)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备：{device}")

    vocab = Vocabulary()
    train_loader, val_loader, _train_dataset, val_dataset = _make_loaders(
        vocab=vocab,
        dataset_type="marker",
        content_len=config.train_content_len,
        train_samples=config.train_samples,
        val_samples=config.val_samples,
        batch_size=config.batch_size,
        answer_offset=config.answer_offset,
        train_seed=config.seed,
        val_seed=2026,
    )
    decode_demo_path = save_decode_demo(
        val_dataset,
        vocab,
        artifact_path("class9_token_decode_demo.txt"),
        num_examples=5,
    )

    model = build_model(vocab, config).to(device)

    print(f"模型参数量：{count_parameters(model)}")
    print("开始训练：目标是预测 <MARK> 后面的数字。")
    result = _train_model(
        model,
        train_loader,
        val_loader,
        device,
        mode="classifier",
        epochs=config.epochs,
        lr=config.lr,
        patience=config.patience,
        min_delta=config.min_delta,
        target_val_acc=config.target_val_acc,
        min_epochs_before_target_stop=config.min_epochs_before_target_stop,
        tag="main",
    )
    model.load_state_dict(result.best_state_dict)
    print(
        f"已恢复 best checkpoint：epoch={result.best_epoch}, "
        f"best_val_loss={result.best_val_loss:.4f}, best_val_acc={result.best_val_acc:.3f}"
    )

    checkpoint_path = save_checkpoint(
        model,
        result,
        artifact_path("class9_transformer_marker_classifier.pt"),
    )
    curve_path = save_training_curves(result.history, artifact_path("class9_transformer_training_curves.png"))

    example_lines, attention_maps, tokens, marker_position, answer_position = collect_prediction_examples(
        model,
        val_dataset,
        vocab,
        device,
        num_examples=8,
    )
    attention_paths = save_attention_diagnostics(
        attention_maps,
        tokens,
        marker_position,
        answer_position,
    )

    print("\n开始训练 Transformer Decoder：目标是自回归预测下一个 token。")
    set_seed(config.seed + 707)
    decoder_train_loader, decoder_val_loader, _decoder_train_dataset, decoder_val_dataset = _make_loaders(
        vocab=vocab,
        dataset_type="decoder",
        content_len=config.decoder_content_len,
        train_samples=config.decoder_train_samples,
        val_samples=config.decoder_val_samples,
        batch_size=config.batch_size,
        train_seed=config.seed + 708,
        val_seed=config.seed + 709,
    )
    decoder_model = build_decoder_model(vocab, config).to(device)
    print(f"Decoder 模型参数量：{count_parameters(decoder_model)}")
    decoder_result = _train_model(
        decoder_model,
        decoder_train_loader,
        decoder_val_loader,
        device,
        mode="decoder",
        epochs=config.decoder_epochs,
        lr=config.decoder_lr,
        patience=config.patience,
        min_delta=config.min_delta,
        target_val_acc=config.target_val_acc,
        min_epochs_before_target_stop=3,
        tag="decoder",
    )
    decoder_model.load_state_dict(decoder_result.best_state_dict)
    decoder_checkpoint_path = save_decoder_checkpoint(
        decoder_model,
        decoder_result,
        artifact_path("class9_transformer_decoder_lm.pt"),
    )
    decoder_curve_path = save_training_curves(
        decoder_result.history,
        artifact_path("class9_decoder_training_curves.png"),
    )
    decoder_report_path, decoder_attention_path = save_decoder_report(
        decoder_model,
        decoder_result,
        decoder_val_dataset,
        vocab,
        device,
        artifact_path("class9_transformer_decoder_report.txt"),
    )

    print("\n开始训练完整 Transformer Encoder-Decoder：目标是 seq2seq copy。")
    set_seed(config.seed + 808)
    seq2seq_train_loader, seq2seq_val_loader, _seq2seq_train_dataset, seq2seq_val_dataset = _make_loaders(
        vocab=vocab,
        dataset_type="seq2seq",
        content_len=config.seq2seq_content_len,
        train_samples=config.seq2seq_train_samples,
        val_samples=config.seq2seq_val_samples,
        batch_size=config.batch_size,
        train_seed=config.seed + 809,
        val_seed=config.seed + 810,
    )
    seq2seq_model = build_seq2seq_model(vocab, config).to(device)
    print(f"Encoder-Decoder 模型参数量：{count_parameters(seq2seq_model)}")
    seq2seq_result = _train_model(
        seq2seq_model,
        seq2seq_train_loader,
        seq2seq_val_loader,
        device,
        mode="seq2seq",
        epochs=config.seq2seq_epochs,
        lr=config.seq2seq_lr,
        patience=config.patience,
        min_delta=config.min_delta,
        target_val_acc=config.target_val_acc,
        min_epochs_before_target_stop=3,
        tag="seq2seq",
    )
    seq2seq_model.load_state_dict(seq2seq_result.best_state_dict)
    seq2seq_checkpoint_path = save_seq2seq_checkpoint(
        seq2seq_model,
        seq2seq_result,
        artifact_path("class9_transformer_seq2seq.pt"),
    )
    seq2seq_curve_path = save_training_curves(
        seq2seq_result.history,
        artifact_path("class9_seq2seq_training_curves.png"),
    )
    seq2seq_report_path, seq2seq_cross_attention_path = save_seq2seq_report(
        seq2seq_model,
        seq2seq_result,
        seq2seq_val_dataset,
        vocab,
        device,
        artifact_path("class9_transformer_seq2seq_report.txt"),
    )

    long_eval = evaluate_long_sequence_generalization(model, vocab, config, device)
    print(
        f"长序列压力测试：content_len={int(long_eval['content_len'])}, "
        f"val_loss={long_eval['val_loss']:.4f}, val_acc={long_eval['val_acc']:.3f}"
    )

    comparison_results = run_comparison_experiments(vocab, config, device)
    comparison_path = save_comparison_plot(
        comparison_results,
        artifact_path("class9_transformer_ablation_comparison.png"),
    )
    report_path = save_report(
        model,
        result,
        example_lines,
        comparison_results,
        long_eval,
        artifact_path("class9_transformer_report.txt"),
    )

    print(f"checkpoint 已保存到：{checkpoint_path}")
    if curve_path is not None:
        print(f"训练曲线已保存到：{curve_path}")
    print(f"decode 示例已保存到：{decode_demo_path}")
    for path in attention_paths:
        print(f"注意力诊断图已保存到：{path}")
    print(f"Decoder checkpoint 已保存到：{decoder_checkpoint_path}")
    if decoder_curve_path is not None:
        print(f"Decoder 训练曲线已保存到：{decoder_curve_path}")
    if decoder_attention_path is not None:
        print(f"Decoder causal 注意力图已保存到：{decoder_attention_path}")
    print(f"Decoder 报告已保存到：{decoder_report_path}")
    print(f"Encoder-Decoder checkpoint 已保存到：{seq2seq_checkpoint_path}")
    if seq2seq_curve_path is not None:
        print(f"Encoder-Decoder 训练曲线已保存到：{seq2seq_curve_path}")
    if seq2seq_cross_attention_path is not None:
        print(f"Encoder-Decoder cross-attention 图已保存到：{seq2seq_cross_attention_path}")
    print(f"Encoder-Decoder 报告已保存到：{seq2seq_report_path}")
    if comparison_path is not None:
        print(f"对比实验图已保存到：{comparison_path}")
    print(f"实验报告已保存到：{report_path}")
    print("第 9 课 artifacts 生成完成。")


if __name__ == "__main__":
    main()
