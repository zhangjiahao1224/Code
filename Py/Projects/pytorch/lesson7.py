# Py\Projects\pytorch 示例：第7课 完整训练 + 保存最佳模型
# 训练 CNN 到高准确率，保存模型参数，并验证单张预测

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 修复 MNIST 下载源
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://mirrors.aliyun.com/pytorch-mnist/mnist/"
]

# 2. 选择计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 3. 数据准备
transform = transforms.ToTensor()  # 仅转换为张量
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. CNN 模型定义（与 lesson5 相同）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. 初始化模型、损失和优化器
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 训练阶段（5轮以获得更高准确率）
print("\n开始训练...")
model.train()
for epoch in range(5):
    total_loss = 0.0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"第 {epoch+1} 轮 损失: {total_loss:.4f}")

# 7. 测试评估
print("\n计算准确率...")
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

acc = 100 * correct / total
print(f"\n✅ 最终准确率: {acc:.2f}%")

# 8. 保存训练好的模型参数
torch.save(model.state_dict(), "mnist_cnn_best.pth")
print("\n✅ 模型已保存：mnist_cnn_best.pth")

# 9. 单张图片验证
img, label = test_dataset[0]  # 取测试集第一张
img = img.unsqueeze(0).to(device)  # 添加 batch 维度
with torch.no_grad():
    _, pred = torch.max(model(img), 1)

print("\n==== 最终测试 ====")
print(f"真实: {label}")
print(f"预测: {pred.item()}")
print("==================")
