import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 设备自动选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力（论文核心）"""
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        # 计算注意力分数: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（padding/未来token掩码）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重 + 加权求和
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. 线性投影 + 拆分多头: (batch_size, num_heads, seq_len, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 缩放点积注意力
        attn_output, attn_weights = ScaledDotProductAttention()(Q, K, V, mask)
        
        # 3. 拼接多头 + 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_O(attn_output)
        return output, attn_weights

class PositionalEncoding(nn.Module):
    """正弦余弦位置编码（论文原版）"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 生成位置编码矩阵: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 不参与训练
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class EncoderLayer(nn.Module):
    """编码器单层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        # 层归一化 + 残差连接
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        # 前馈网络层
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class DecoderLayer(nn.Module):
    """解码器单层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 解码器自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads) # 编码器-解码器注意力
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. 解码器自注意力（带未来token掩码）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 2. 编码器-解码器注意力
        cross_attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # 3. 前馈网络层
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x

class Transformer(nn.Module):
    """完整Transformer架构（论文原版）"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        # 编码器/解码器堆叠
        self.encoder = nn.Sequential(*[EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.Sequential(*[DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        # 输出层
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        # 初始化参数（论文推荐）
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src/tgt: (batch_size, seq_len)
        # 词嵌入 + 位置编码（乘以sqrt(d_model)是论文推荐操作）
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim))
        tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim))
        
        # 编码器前向
        enc_output = self.encoder(src_emb, src_mask)
        # 解码器前向
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        # 输出层
        output = self.fc(dec_output)
        return output