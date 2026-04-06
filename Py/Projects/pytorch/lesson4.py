# Py\Projects\pytorch 示例：MNIST 模型保存与加载
# 本示例展示：训练、保存模型参数、重新加载模型并做推断

import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 1. 数据准备阶段
# 当国内访问 MNIST 网站稳定性较差时，预设备用 mirror
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://mirrors.aliyun.com/pytorch-mnist/mnist/"
]

# 自动选择设备：CUDA>CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 数据转换：将图片转为张量，值从 [0,255] 归一化为 [0,1]
transform = transforms.Compose([transforms.ToTensor()])

# 只加载测试集，用于后续示例预测（批量为1）
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# 2. 模型定义（结构与训练时一致）
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 两层简易全连接网络
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 展平输入形状： [batch, 1, 28, 28] -> [batch, 784]
        x = x.view(-1, 28*28)
        # 隐含层激活
        x = torch.relu(self.fc1(x))
        # 输出 logits（未归一化）
        x = self.fc2(x)
        return x

# 3. 创建模型 + 训练设置
model = Model().to(device)
criterion = nn.CrossEntropyLoss()  # 适用于多分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练阶段（本课简化为1轮）
print("\n开始快速训练...")
model.train()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(1):
    total_loss = 0.0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()  # 清零梯度
        output = model(data)
        loss = criterion(output, label)
        loss.backward()        # 反向传播
        optimizer.step()       # 参数更新

        total_loss += loss.item()

    print(f"训练损失: {total_loss:.4f}")

# 5. 保存模型参数（state_dict 格式）
print("\n✅ 保存模型到 mnist_model.pth")
torch.save(model.state_dict(), "mnist_model.pth")

# 6. 加载模型参数
print("✅ 加载已保存的模型")
model_load = Model().to(device)
# 重要：load_state_dict 需要与保存时结构一致
model_load.load_state_dict(torch.load("mnist_model.pth"))
model_load.eval()  # 评估模式，关闭 dropout/batchnorm 训练状态

# 7. 使用加载好的模型做推断
print("\n开始预测测试图片：")
with torch.no_grad():  # 关闭梯度计算，提高推断速度
    for i, (img, label) in enumerate(test_loader):
        img, label = img.to(device), label.to(device)
        output = model_load(img)
        _, pred = torch.max(output, 1)

        print(f"真实数字: {label.item()}, 模型预测: {pred.item()}")

        if i == 5:  # 只展示前6张
            break

print("\n🎉 第4课完成：你学会了保存 & 加载 & 部署模型！")