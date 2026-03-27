# Py\Projects\pytorch
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 依旧修复MNIST下载
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://mirrors.aliyun.com/pytorch-mnist/mnist/"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 数据预处理
transform = transforms.Compose([transforms.ToTensor()])

# 加载测试集（这节课我们只演示保存和预测，训练可以简写）
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# 模型定义（必须和训练时一模一样！）
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 1. 创建模型并训练（简化版，快速训练一遍）
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 快速训练1轮
print("\n开始快速训练...")
model.train()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(1):
    total_loss = 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"训练损失: {total_loss:.4f}")

# ==================== 第4课重点：模型保存 ====================
print("\n✅ 保存模型到 mnist_model.pth")
torch.save(model.state_dict(), "mnist_model.pth")

# ==================== 第4课重点：模型加载 ====================
print("✅ 加载已保存的模型")
model_load = Model().to(device)
model_load.load_state_dict(torch.load("mnist_model.pth"))
model_load.eval()  # 切换到评估模式

# ==================== 使用加载好的模型做预测 ====================
print("\n开始预测测试图片：")
with torch.no_grad():
    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        output = model_load(img)
        _, pred = torch.max(output, 1)
        
        print(f"真实数字: {label.item()}, 模型预测: {pred.item()}")
        
        if i == 5:  # 只展示5张
            break

print("\n🎉 第4课完成：你学会了保存 & 加载 & 部署模型！")