# Py\Projects\pytorch
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image

# 修复数据集下载
datasets.MNIST.mirrors = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://mirrors.aliyun.com/pytorch-mnist/mnist/"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ==================== CNN 模型 ====================
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

# 加载模型
model = CNN().to(device)
model.eval()

# ==================== 从测试集里拿一张图片直接预测 ====================
transform = transforms.ToTensor()
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 直接取第 0 张图片
img, label = test_dataset[0]
img = img.unsqueeze(0).to(device)  # 变成模型需要的格式

# 预测
with torch.no_grad():
    output = model(img)
    _, pred = torch.max(output, 1)

# 输出结果
print("\n===== 预测结果 =====")
print(f"真实数字 : {label}")
print(f"模型预测 : {pred.item()}")
print("===================")

print("\n🎉 Lesson6 运行成功！")