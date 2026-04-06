# Py\Projects\pytorch 示例：第9课 CIFAR10 图像分类部署
# 加载预训练 CNN 模型，对用户提供的图片进行 10 类分类预测

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. CIFAR10 数据集 10 类标签（中文翻译）
classes = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]

# 2. 选择计算设备（优先 GPU）
device = torch.device("cuda")
print("使用设备:", device)

# 3. CNN 模型定义（用于 CIFAR10 分类）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1：3通道 RGB -> 16通道，3x3卷积，padding=1
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 卷积层2：16 -> 32通道，3x3卷积，padding=1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1：32*8*8 -> 256（两次池化后 32x32 -> 8x8）
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        # 全连接层2：256 -> 128
        self.fc2 = nn.Linear(256, 128)
        # 全连接层3：128 -> 10类
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 前向传播：卷积 + ReLU + 池化
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # 展平（保留批次维度）
        x = torch.flatten(x, 1)
        # 全连接 + ReLU
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. 初始化模型并加载预训练权重
try:
    model = Net().to(device)
    model.load_state_dict(torch.load("cifar10_model.pth", weights_only=True))
except FileNotFoundError:
    print("❌ 错误：找不到 'cifar10_model.pth' 文件。请确保模型已训练并保存。")
    exit(1)
model.eval()  # 评估模式，关闭 dropout/batchnorm 训练行为

# 5. 预测时的数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),                    # 缩放到 CIFAR10 输入尺寸
    transforms.ToTensor(),                          # 转为张量 [0,1]
    transforms.Normalize((0.5, 0.5, 0.5),           # 归一化到 [-1,1]（RGB 各通道）
                         (0.5, 0.5, 0.5))
])

# 6. 单张图片预测函数
def predict(img_path):
    try:
        img = Image.open(img_path).convert("RGB")      # 加载并转为 RGB
    except FileNotFoundError:
        print(f"❌ 错误：找不到图片文件 '{img_path}'。请检查路径。")
        return None, None
    img = transform(img).unsqueeze(0).to(device)    # 预处理 + 添加 batch 维度

    with torch.no_grad():                           # 关闭梯度，加速推理
        out = model(img)                            # 前向计算 logits [1,10]
        idx = out.argmax().item()                   # 取最大值索引
        prob = torch.softmax(out, dim=1)[0][idx].item() * 100  # 计算置信度

    return classes[idx], prob

# 7. 测试示例（替换为你的图片路径）
res, conf = predict("car.jpg")
if res is not None:
    print(f"\n🎯 预测：{res}")
    print(f"🎯 置信度：{conf:.2f}%")