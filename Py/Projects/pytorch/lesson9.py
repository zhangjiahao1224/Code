# Py\Projects\pytorch
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# CIFAR10 10 类标签（中文）
classes = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]

# 设备选择：优先 cuda，如果不可用请改为 torch.device('cpu')
device = torch.device("cuda")
print("使用设备:", device)

# 定义 CNN 模型，包含两层卷积 + 两次池化 + 三层全连接
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 3 通道 RGB -> 16 通道
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 16 -> 32 通道
        self.pool = nn.MaxPool2d(2, 2)               # 池化尺寸 2x2
        # 两次池化后，32x32 -> 16x16 -> 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)                 # 最终 10 类

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))     # 卷积 + ReLU + 池化
        x = self.pool(torch.relu(self.conv2(x)))     # 卷积 + ReLU + 池化
        x = torch.flatten(x, 1)                      # 展平（保留批次维）
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化模型并加载预训练参数（来自 train.py 训练）
model = Net().to(device)
model.load_state_dict(torch.load("cifar10_model.pth", weights_only=True))
model.eval()  # 切换为评估模式

# 预测时的图像预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),                    # CIFAR10 输入尺寸
    transforms.ToTensor(),                          # 转为 Tensor
    transforms.Normalize((0.5, 0.5, 0.5),           # 归一化到 [-1, 1]
                         (0.5, 0.5, 0.5))
])

# 预测单张图片的函数
def predict(img_path):
    img = Image.open(img_path).convert("RGB")      # 转为 RGB
    img = transform(img).unsqueeze(0).to(device)    # 增加 batch 维度

    with torch.no_grad():                           # 关闭梯度计算，加速推理
        out = model(img)
        idx = out.argmax().item()                   # 最大 logit 对应类别索引
        prob = torch.softmax(out, dim=1)[0][idx].item() * 100  # 置信度

    return classes[idx], prob

# 测试实例：将当前目录下的 car.jpg 换成你自己的样本图像
res, conf = predict("car.jpg")
print(f"\n🎯 预测：{res}")
print(f"🎯 置信度：{conf:.2f}%")