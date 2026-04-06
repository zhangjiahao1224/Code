# Py\Projects\pytorch 示例：第6课 模型部署与单张图片预测
# 演示：加载预训练模型，对单张 MNIST 图片进行预测

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

# 1. 修复 MNIST 下载源（国内环境）
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://mirrors.aliyun.com/pytorch-mnist/mnist/"
]

# 2. 选择计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ==================== 3. CNN 模型定义 ====================
# 与 lesson5 相同的 CNN 结构，用于 MNIST 分类
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1：1通道输入 -> 32通道输出，3x3卷积，padding=1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层2：32 -> 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层1：展平后 64*7*7 -> 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2：128 -> 10类输出
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播：卷积 -> 激活 -> 池化 -> 卷积 -> 激活 -> 池化
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # 展平为全连接输入
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. 初始化模型并切换到评估模式
model = CNN().to(device)
model.eval()  # 评估模式，关闭 dropout/batchnorm 训练行为

# ==================== 5. 单张图片预测演示 ====================
# 数据转换：仅转为张量
transform = transforms.ToTensor()
# 加载测试集（无需训练集）
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 从测试集中取第0张图片（真实标签为5）
img, label = test_dataset[0]
# 添加 batch 维度： [1,28,28] -> [1,1,28,28]
img = img.unsqueeze(0).to(device)

# 6. 进行预测
with torch.no_grad():  # 关闭梯度计算，提高速度
    output = model(img)  # 输出 logits [1,10]
    _, pred = torch.max(output, 1)  # 取最大值索引作为预测类别

# 7. 输出结果
print("\n===== 预测结果 =====")
print(f"真实数字 : {label}")
print(f"模型预测 : {pred.item()}")
print("===================")

print("\n🎉 Lesson6 运行成功！")
# 注意：此模型未加载预训练权重，预测结果随机。实际部署时需加载 state_dict。