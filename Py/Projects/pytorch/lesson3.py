# Py\Projects\pytorch 示例：MNIST 手写数字识别（全流程）
# 包含：数据准备、模型定义、训练、测试、单样本预测

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. 数据准备
# ----------------------

# 修复 MNIST 下载 URL，避免在国内环境下载失败
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://mirrors.aliyun.com/pytorch-mnist/mnist/"
]

# 自动选择计算设备：优先 GPU，否则 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 数据转换：将 PIL 图像转换为 [0,1] 的张量
transform = transforms.Compose([transforms.ToTensor()])

# 下载 MNIST 训练集和测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoader 封装批次迭代器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ----------------------
# 2. 模型定义
# ----------------------

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 28x28 图像展平为 784 维，第一层到 128 隐含节点
        self.fc1 = nn.Linear(28*28, 128)
        # 输出层 10 类（0-9）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: [batch, 1, 28, 28] -> [batch, 784]
        x = x.view(-1, 28*28)
        # 激活函数使用 ReLU
        x = torch.relu(self.fc1(x))
        # 输出 logits
        x = self.fc2(x)
        return x

# 实例化模型并移动至设备
model = Model().to(device)

# 损失函数：交叉熵（内部包含 softmax）
criterion = nn.CrossEntropyLoss()
# 优化器：Adam，学习率 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------
# 3. 训练
# ----------------------
print("\n开始训练...")
model.train()  # 训练模式启用 Dropout/BatchNorm 训练行为（本例无）

for epoch in range(3):  # 训练 3 轮
    total_loss = 0.0
    for data, label in train_loader:
        # 送入设备
        data, label = data.to(device), label.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向计算 + 反向传播
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"第 {epoch+1} 轮，损失: {total_loss:.4f}")

# ----------------------
# 4. 测试/评估
# ----------------------
print("\n在测试集上评估准确率...")
correct = 0
total = 0
model.eval()  # 评估模式，关闭梯度计算与训练行为

with torch.no_grad():
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)

        # 取最大值所在索引作为预测类别
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = 100 * correct / total
print(f"测试集准确率: {accuracy:.2f}%")

# ----------------------
# 5. 单样本演示
# ----------------------
# 从测试集中取一张图片（不随机）
images, labels = next(iter(test_loader))
img = images[0].to(device)
label = labels[0].to(device)

# 预测单张图像
with torch.no_grad():
    output = model(img.unsqueeze(0))  # 模型需要 batch 维度
    _, pred = torch.max(output, 1)

print(f"\n模型预测数字是: {pred.item()}, 真实标签是: {label.item()}")
print("\n训练全部完成！🎉")