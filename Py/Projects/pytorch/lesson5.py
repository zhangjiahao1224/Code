# Py\Projects\pytorch 示例：第5课 CNN（卷积神经网络）
# 训练 MNIST 手写数字识别模型并计算准确率

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 修复 MNIST 下载源
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://mirrors.aliyun.com/pytorch-mnist/mnist/"
]

# 2. 选择计算设备：cuda > cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 3. 数据预处理：仅转换为张量（0~1）
transform = transforms.Compose([transforms.ToTensor()])

# 4. 准备数据集及迭代器
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==================== 第5课重点：CNN 卷积神经网络 ====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层：1->32 个 3x3 卷积，padding=1 保持尺寸不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 池化层：2x2 最大池化，尺寸减半
        self.pool = nn.MaxPool2d(2, 2)
        # 第二层：32->64 个卷积
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 全连接层：8x8 (经过两次 2x2 池化后 28->14->7) -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 输出层：10 类
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 输入 x 形状 [batch, 1, 28, 28]
        x = self.pool(torch.relu(self.conv1(x)))  # [batch,32,14,14]
        x = self.pool(torch.relu(self.conv2(x)))  # [batch,64,7,7]
        
        # 展平为全连接层输入
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. 初始化模型、损失和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # 分类损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 训练
print("\n开始训练 CNN 模型...")
model.train()
for epoch in range(2):  # 简化示例只跑小轮数
    total_loss = 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"第 {epoch+1} 轮，损失: {total_loss:.4f}")

# 7. 测试评估
print("\n测试集准确率计算中...")
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

accuracy = 100 * correct / total
print(f"\n🎉 CNN 模型准确率: {accuracy:.2f}%")
print("第5课完成！你学会了卷积神经网络！")