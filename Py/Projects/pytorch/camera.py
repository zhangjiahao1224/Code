# Py\Projects\pytorch - CIFAR10 实时摄像头分类
import torch
import torch.nn as nn
import cv2
import torchvision.transforms as transforms
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# CIFAR10 类别标签（中文）
classes = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]

# 每个类别对应不同颜色（用于绘制识别框）
colors = [
    (255,0,0),    # 飞机 - 蓝
    (0,255,0),    # 汽车 - 绿
    (0,0,255),    # 鸟 - 红
    (255,255,0),  # 猫 - 青
    (255,0,255),  # 鹿 - 紫
    (0,255,255),  # 狗 - 黄
    (128,0,128),  # 青蛙 - 深紫
    (255,165,0),  # 马 - 橙
    (0,128,128),  # 船 - 深青
    (128,128,0)   # 卡车 - 橄榄
]

# 设备选择：优先 GPU
device = torch.device("cuda")

# ==================== CIFAR10 CNN 模型定义 ====================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一卷积层：3通道(RGB)->16通道，3x3卷积，padding=1保持尺寸
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 第二卷积层：16->32通道
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 池化层：2x2最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 两次池化后，32x32 -> 16x16 -> 8x8
        # 全连接层
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10类输出

    def forward(self, x):
        # 前向传播：卷积 -> ReLU -> 池化 -> 卷积 -> ReLU -> 池化 -> 展平 -> 全连接
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==================== 加载训练好的模型 ====================
model = Net().to(device)
# 加载权重（确保 train.py 已运行生成 cifar10_model.pth）
model.load_state_dict(torch.load("cifar10_model.pth", weights_only=True))
model.eval()  # 设置为评估模式

# ==================== 图像预处理 ====================
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量 [0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1,1]
])

# ==================== 打开摄像头 ====================
cap = cv2.VideoCapture(0)  # 0表示默认摄像头
print("✅ 摄像头已启动！按 Q 退出！")

# ==================== 加载中文字体 ====================
# 使用 Windows 系统字体，确保中文显示无乱码
font = ImageFont.truetype("simsun.ttc", 40)

# ==================== 实时预测循环 ====================
while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法打开摄像头")
        break

    # 获取帧尺寸
    h, w = frame.shape[:2]
    
    # 缩放帧到模型输入尺寸 (32x32)
    img = cv2.resize(frame, (32, 32))
    # 应用预处理变换
    img = transform(img).unsqueeze(0).to(device)  # 增加批次维度，移到设备

    # ==================== 模型推理 ====================
    with torch.no_grad():  # 关闭梯度计算，加速推理
        out = model(img)
        idx = out.argmax().item()  # 获取预测类别索引
        prob = torch.softmax(out, dim=1)[0][idx].item() * 100  # 计算置信度

    # 构建显示标签
    label = f"{classes[idx]} {prob:.1f}%"  # 例如："汽车 87.5%"
    color = colors[idx]  # 根据类别选择颜色

    # -------------------------- 绘制识别框 --------------------------
    # 在画面中央绘制一个大框，表示识别区域
    x1, y1 = int(w*0.2), int(h*0.2)  # 左上角
    x2, y2 = int(w*0.8), int(h*0.8)  # 右下角
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # 绘制彩色框

    # -------------------------- 绘制中文标签 --------------------------
    # 将 OpenCV 图像转为 PIL 图像，以便绘制中文
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    # 在框上方绘制类别和置信度
    draw.text((x1, y1-50), label, font=font, fill=color)
    # 转回 OpenCV 格式
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # ==================== 显示结果 ====================
    cv2.imshow("AI实时识别", frame)

    # ==================== 退出条件 ====================
    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==================== 清理资源 ====================
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 关闭所有窗口

print("✅ 实时识别结束！")