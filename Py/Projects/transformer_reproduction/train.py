import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import math

from model import Transformer, DEVICE
from data_utils import (
    load_wmt14_dataset, create_data_loader, create_mask,
    TOKENIZER, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
)

# 超参数（论文原版）
CONFIG = {
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 2048,
    "batch_size": 64,
    "epochs": 20,
    "lr": 0.0001,
    "warmup_steps": 4000,
    "dropout": 0.1
}

def lr_scheduler(step, d_model=512, warmup_steps=4000):
    """论文原版学习率调度器"""
    step = max(1, step)
    return (d_model ** (-0.5)) * min(step ** (-0.5), step * (warmup_steps ** (-1.5)))

def train_epoch(model, train_loader, criterion, optimizer, step):
    """单轮训练"""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for src, tgt in progress_bar:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        src_mask, tgt_mask = create_mask(src, tgt[:, :-1])  # 解码器输入去掉最后一个token
        
        # 前向传播
        output = model(src, tgt[:, :-1], src_mask, tgt_mask)
        
        # 计算损失（标签去掉第一个token）
        loss = criterion(
            output.reshape(-1, output.size(-1)),
            tgt[:, 1:].reshape(-1)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪（论文推荐）
        
        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_scheduler(step)
        
        # 更新参数
        optimizer.step()
        step += 1
        
        # 记录损失
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, step

@torch.no_grad()
def evaluate(model, val_loader):
    """评估（BLEU分数）"""
    model.eval()
    bleu = BLEU()
    predictions = []
    references = []
    progress_bar = tqdm(val_loader, desc="Evaluating")
    
    for src, tgt in progress_bar:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        # 贪心解码生成目标序列
        batch_size = src.size(0)
        # 初始化解码器输入（仅<SOS>标记）
        tgt_pred = torch.ones((batch_size, 1), dtype=torch.long, device=DEVICE) * SOS_TOKEN
        
        # 自回归生成（最大长度50）
        for _ in range(50):
            src_mask, tgt_mask = create_mask(src, tgt_pred)
            output = model(src, tgt_pred, src_mask, tgt_mask)
            # 取最后一个token的预测
            next_token = output.argmax(-1)[:, -1].unsqueeze(1)
            tgt_pred = torch.cat([tgt_pred, next_token], dim=1)
            # 终止条件：所有样本都生成<EOS>
            if (next_token == EOS_TOKEN).all():
                break
        
        # 转文本
        for i in range(batch_size):
            # 解码预测结果（去掉<SOS>/<EOS>）
            pred_ids = tgt_pred[i].tolist()
            pred_ids = pred_ids[1: pred_ids.index(EOS_TOKEN) if EOS_TOKEN in pred_ids else -1]
            pred_text = TOKENIZER.decode(pred_ids, skip_special_tokens=True)
            
            # 解码参考结果
            ref_ids = tgt[i].tolist()
            ref_ids = ref_ids[1: ref_ids.index(EOS_TOKEN) if EOS_TOKEN in ref_ids else -1]
            ref_text = TOKENIZER.decode(ref_ids, skip_special_tokens=True)
            
            predictions.append(pred_text)
            references.append([ref_text])
    
    # 计算BLEU分数
    bleu_score = bleu.corpus_score(predictions, references).score
    return bleu_score

def main():
    # 1. 加载数据
    dataset = load_wmt14_dataset()
    train_loader = create_data_loader(dataset["train"], batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = create_data_loader(dataset["validation"], batch_size=32, shuffle=False)
    
    # 2. 初始化模型
    vocab_size = TOKENIZER.vocab_size
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        d_ff=CONFIG["d_ff"],
        dropout=CONFIG["dropout"]
    ).to(DEVICE)
    
    # 3. 优化器与损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)  # 忽略padding损失
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        betas=(0.9, 0.98),  # 论文原版参数
        eps=1e-9
    )
    
    # 4. 训练循环
    step = 0
    best_bleu = 0.0
    for epoch in range(CONFIG["epochs"]):
        print(f"\n===== Epoch {epoch+1}/{CONFIG['epochs']} =====")
        
        # 训练
        train_loss, step = train_epoch(model, train_loader, criterion, optimizer, step)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 评估
        bleu_score = evaluate(model, val_loader)
        print(f"Validation BLEU Score: {bleu_score:.2f}")
        
        # 保存最佳模型
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_bleu": best_bleu
            }, "best_transformer.pth")
            print(f"保存最佳模型 (BLEU: {best_bleu:.2f})")
    
    print(f"\n训练完成！最佳BLEU分数: {best_bleu:.2f}")

if __name__ == "__main__":
    main()