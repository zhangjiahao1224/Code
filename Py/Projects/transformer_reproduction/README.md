# Transformer复现（Attention Is All You Need）
基于PyTorch复现《Attention Is All You Need》中的Transformer模型，适配WMT14英德翻译任务。

## 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

## 快速运行
# 开始训练（自动下载数据集+训练+评估）
python train.py