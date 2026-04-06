# Py\Projects\pytorch 示例：第8课 模型部署 - 预测用户手写数字
# 加载预训练 CNN 模型，对用户提供的图片进行预测

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import PIL.ImageOps

# 1. 选择计算设备（强制 GPU）
device = torch.device("cuda")
print("使用设备:", device)

# 2. CNN 模型定义（与训练时一致）
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1：1->32，3x3，padding=1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # 卷积层2：32->64，3x3，padding=1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(2)
        # 全连接层1：64*7*7 -> 128
        self.fc1 = nn.Linear(64*7*7, 128)
        # 全连接层2：128 -> 10类
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. 初始化模型并加载预训练权重
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn_best.pth", weights_only=True))
model.eval()  # 评估模式

# 4. 数据预处理（针对用户手写图片）
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 缩放到 28x28
    transforms.ToTensor(),        # 转为张量 [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 标准化
])

# 5. 加载用户图片并预处理
try:
    img = Image.open("Py/Projects/pytorch/test.png").convert("L")  # 转为灰度
except FileNotFoundError:
    print("❌ 错误：找不到 'Py/Projects/pytorch/test.png' 文件。请确保图片路径正确。")
    exit(1)
img = PIL.ImageOps.invert(img)             # 反色（黑底白字 -> 白底黑字）
img = transform(img).unsqueeze(0).to(device)  # 添加 batch 维度，送至设备

# 6. 进行预测
with torch.no_grad():
    output = model(img)  # 输出 logits [1,10]
    prob = torch.softmax(output, dim=1)  # 转为概率
    pred = output.argmax().item()  # 预测类别
    conf = prob[0][pred].item() * 100  # 置信度百分比

print(f"\n🎯 你画的数字：{pred}")
print(f"🎯 置信度：{conf:.2f}%")