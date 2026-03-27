# Py\Projects\pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设备（优先GPU，否则CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练使用设备:", device)

# 数据预处理：
# 1) Resize：统一到 32x32
# 2) ToTensor：转为 [0,1] 张量
# 3) Normalize：按通道归一化到 [-1,1]
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR10 训练集，第一次会下载数据到 ./data
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# DataLoader 提供批量、打乱、并行加载
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# 模型：简单两层卷积 + 三层全连接
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1：3通道（RGB）->16通道，3x3卷积，padding=1保持尺寸
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # Conv2：16通道->32通道
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 最大池化，尺寸减半
        # 两次池化后，32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10个类别

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # C1
        x = self.pool(torch.relu(self.conv2(x)))  # C2
        x = torch.flatten(x, 1)  # 展平为向量，保留 batch 维
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net().to(device)  # 送入设备

# 损失函数 + 优化器
# CrossEntropyLoss 适用于 10 类分类（logits -> 交叉熵）
criterion = nn.CrossEntropyLoss()
# Adam 优化器（自适应学习率），lr=0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 开始训练
print("\n开始训练...")
for epoch in range(10):  # 训练 10 轮
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()      # 清空梯度
        outputs = model(inputs)    # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()            # 反向传播
        optimizer.step()           # 更新参数

        running_loss += loss.item()

    avg_loss = running_loss / len(trainloader)
    print(f"第 {epoch+1} 轮，损失: {avg_loss:.3f}")

# 保存训练好的模型
torch.save(model.state_dict(), "cifar10_model.pth")
print("\n✅ 训练完成！模型已保存为 cifar10_model.pth")