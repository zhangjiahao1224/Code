import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 预训练分词器（替代论文BPE，简化复现）
TOKENIZER = AutoTokenizer.from_pretrained("t5-small", src_lang="de", tgt_lang="en")
SOS_TOKEN = TOKENIZER.bos_token_id  # 开始标记
EOS_TOKEN = TOKENIZER.eos_token_id  # 结束标记
PAD_TOKEN = TOKENIZER.pad_token_id  # 填充标记

def load_wmt14_dataset():
    """加载WMT14英德翻译数据集（论文同款）"""
    print("加载WMT14英德翻译数据集...")
    dataset = load_dataset("wmt14", "de-en", split={"train": "train", "val": "validation"})
    return dataset

def preprocess_function(examples):
    """预处理函数：分词+添加首尾标记"""
    src_texts = [ex["de"] for ex in examples["translation"]]
    tgt_texts = [ex["en"] for ex in examples["translation"]]
    
    # 分词
    src_encoded = TOKENIZER(src_texts, truncation=True, max_length=128, return_attention_mask=False)
    tgt_encoded = TOKENIZER(tgt_texts, truncation=True, max_length=128, return_attention_mask=False)
    
    # 添加<SOS>/<EOS>标记
    src_ids = [ids for ids in src_encoded["input_ids"]]
    tgt_ids = [[SOS_TOKEN] + ids + [EOS_TOKEN] for ids in tgt_encoded["input_ids"]]
    
    return {"src_ids": src_ids, "tgt_ids": tgt_ids}

def create_mask(src, tgt):
    """生成掩码：src_mask（padding掩码） + tgt_mask（padding+未来token掩码）"""
    # 1. 源序列掩码: (batch_size, 1, 1, src_seq_len)
    src_mask = (src != PAD_TOKEN).unsqueeze(1).unsqueeze(2).to(DEVICE)
    
    # 2. 目标序列掩码
    tgt_seq_len = tgt.size(1)
    # padding掩码
    tgt_pad_mask = (tgt != PAD_TOKEN).unsqueeze(1).unsqueeze(2).to(DEVICE)
    # 未来token掩码（下三角矩阵）
    tgt_no_peak_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=DEVICE)).bool()
    # 组合掩码
    tgt_mask = tgt_pad_mask & tgt_no_peak_mask
    
    return src_mask, tgt_mask

def collate_fn(batch):
    """数据批处理函数：padding+生成掩码"""
    src_ids = [torch.tensor(item["src_ids"], dtype=torch.long) for item in batch]
    tgt_ids = [torch.tensor(item["tgt_ids"], dtype=torch.long) for item in batch]
    
    # padding到批次最大长度
    src_padded = pad_sequence(src_ids, batch_first=True, padding_value=PAD_TOKEN)
    tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=PAD_TOKEN)
    
    return src_padded, tgt_padded

def create_data_loader(dataset, batch_size=64, shuffle=True):
    """创建数据加载器"""
    # 预处理数据集
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 创建DataLoader
    data_loader = DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    
    return data_loader

# 测试数据加载
if __name__ == "__main__":
    dataset = load_wmt14_dataset()
    train_loader = create_data_loader(dataset["train"], batch_size=2)
    
    # 测试批次
    for src, tgt in train_loader:
        src_mask, tgt_mask = create_mask(src, tgt)
        print("源序列形状:", src.shape)
        print("目标序列形状:", tgt.shape)
        print("源掩码形状:", src_mask.shape)
        print("目标掩码形状:", tgt_mask.shape)
        break