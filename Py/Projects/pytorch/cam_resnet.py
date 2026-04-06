# ResNet + CAM 实时摄像头 CIFAR10 识别
# 使用预训练 ResNet18 模型进行实时物体识别，包含置信度过滤和中文显示

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 OpenMP 库冲突

import torch
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from torchvision import models, transforms

# ----------------------
# 1. GPU 加速配置（针对 RTX 4090）
# ----------------------
device = torch.device("cuda")  # 强制使用 GPU
torch.backends.cudnn.benchmark = True  # 启用 cuDNN 自动调优，提高推理速度

# 2. CIFAR10 数据集类别标签（中文）
classes = [
    "飞机","汽车","鸟","猫","鹿",
    "狗","青蛙","马","船","卡车"
]
# 类别对应颜色（用于绘制边框）
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
          (0,255,255),(128,0,128),(255,165,0),(0,128,128),(128,128,0)]

# ----------------------
# 3. 模型加载与配置
# ----------------------
model = models.resnet18()  # 加载预训练 ResNet18 骨干
model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 修改输出层为 10 类
model.load_state_dict(torch.load("resnet_cifar10.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()  # 评估模式，关闭 dropout/batchnorm

# 4. 图像预处理（与训练时对齐）
trans = transforms.Compose([
    transforms.Resize((224,224)),  # ResNet 输入尺寸
    transforms.ToTensor(),          # 转为张量
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])  # ImageNet 标准化
])

# ----------------------
# 5. 摄像头初始化
# ----------------------
cap = cv2.VideoCapture(0)  # 打开默认摄像头
# 解决中文显示乱码：使用系统宋体
font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", 36, encoding="utf-8")
print("✅ ResNet 4090 实时识别启动 | 按 Q 退出")

# ----------------------
# 6. 主推理循环
# ----------------------
with torch.no_grad():  # 关闭梯度计算，加速推理
    while True:
        ret, frame = cap.read()  # 读取摄像头帧
        if not ret: break        # 如果读取失败，退出
        h, w = frame.shape[:2]   # 获取帧尺寸

        # 图像预处理 + GPU 推理
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # BGR -> RGB
        img_tensor = trans(pil_img).unsqueeze(0).to(device)  # 预处理 + 添加 batch 维度
        out = model(img_tensor)  # 前向推理
        idx = out.argmax().item()  # 取最大概率类别索引
        conf = torch.softmax(out, dim=1)[0][idx].item() * 100  # 计算置信度

        # 🔥 置信度过滤：低于 70% 显示为未知物体
        if conf < 70:
            text = f"未知物体 {conf:.1f}%"
            box_color = (128, 128, 128)  # 灰色边框
        else:
            text = f"{classes[idx]} {conf:.1f}%"
            box_color = colors[idx]  # 类别对应颜色

        # 绘制识别框和中文标签
        cv2.rectangle(frame, (50, 50), (w-50, h-50), box_color, 3)  # 画边框
        frame_pil = Image.fromarray(frame)  # 转为 PIL 图像以绘制中文
        draw = ImageDraw.Draw(frame_pil)
        draw.text((60, 20), text, font=font, fill=box_color)  # 绘制文本
        frame = np.array(frame_pil)  # 转回 numpy 数组

        # 显示窗口（标题用英文避免乱码）
        cv2.imshow("ResNet Real-time Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 按 Q 退出
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()