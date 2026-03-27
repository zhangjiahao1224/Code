# Py\Projects\pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 修复数据集下载
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://mirrors.aliyun.com/pytorch-mnist/mnist/"
]

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])

# 加载数据
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==================== 第5课重点：CNN 卷积神经网络 ====================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层 + 池化层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
print("\n开始训练 CNN 模型...")
model.train()
for epoch in range(2):
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

# 测试准确率
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

print(f"\n🎉 CNN 模型准确率: {100 * correct / total:.2f}%")
print("第5课完成！你学会了卷积神经网络！")