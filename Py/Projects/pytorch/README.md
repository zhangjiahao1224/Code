# PyTorch 项目集合

这是一个 PyTorch 学习和实践的项目集合，涵盖了从基础 CNN 到高级目标检测的多个示例。适合初学者逐步学习深度学习。

## 项目结构

```
pytorch/
├── README.md              # 项目说明文档
├── lesson1.py             # PyTorch 基础：张量与 GPU
├── lesson2.py             # 线性回归：神经网络拟合
├── lesson3.py             # MNIST 训练：全连接网络
├── lesson4.py             # 模型保存与加载
├── lesson5.py             # CNN 基础：卷积神经网络
├── lesson6.py             # MNIST 推理（预训练模型）
├── lesson7.py             # MNIST 训练（CNN）
├── lesson8.py             # MNIST 本地图片预测
├── train.py               # CIFAR10 训练
├── lesson9.py             # CIFAR10 本地图片预测
├── camera.py              # CIFAR10 实时摄像头分类
├── yolo.py                # YOLOv8 实时检测
├── data/                  # 数据集下载目录（自动创建）
├── car.jpg                # CIFAR10 示例图片
├── test.png               # MNIST 示例图片
├── mnist_cnn_best.pth     # MNIST CNN 模型
├── mnist_model.pth        # MNIST 全连接模型
├── cifar10_model.pth      # CIFAR10 模型
└── yolov8n.pt             # YOLOv8 Nano 模型
```

## 详细文件说明

### PyTorch 基础教程

- **`lesson1.py`**:
  - 功能：PyTorch 环境检查、张量创建、GPU 使用基础
  - 输出：PyTorch 版本、GPU 可用性、张量操作结果
  - 示例输出：
    ```
    PyTorch版本: 2.0.1
    GPU是否可用: True
    你的显卡: NVIDIA GeForce RTX 3060
    ```

- **`lesson2.py`**:
  - 功能：使用神经网络进行线性回归拟合
  - 模型：单层线性网络
  - 输出：训练损失 + 学到的参数
  - 示例输出：
    ```
    AI 学到的公式：y = 1.98x + 0.05
    本来的公式：y = 2x + 0
    ```

- **`lesson3.py`**:
  - 功能：使用全连接网络训练 MNIST 手写数字识别
  - 模型：2 层全连接网络
  - 训练：3 轮，批次大小 64
  - 输出：训练损失 + 测试准确率

- **`lesson4.py`**:
  - 功能：模型保存与加载演示
  - 保存：训练后保存为 `mnist_model.pth`
  - 加载：重新加载模型进行预测
  - 输出：保存确认 + 预测结果

- **`lesson5.py`**:
  - 功能：CNN 卷积神经网络训练 MNIST
  - 模型：2 层卷积 + 2 层全连接
  - 特点：介绍卷积、池化操作
  - 输出：训练损失 + 准确率

### 基础 MNIST 手写数字识别 (0-9)

- **`lesson6.py`**:
  - 功能：加载预训练 CNN 模型，对 MNIST 测试集单张图片进行推理。
  - 输入：无（使用内置测试集）
  - 输出：预测数字 + 真实标签
  - 示例输出：
    ```
    使用设备: cuda
    ===== 预测结果 =====
    真实数字 : 5
    模型预测 : 5
    ====================
    ```

- **`lesson7.py`**:
  - 功能：从头训练 CNN 模型识别 MNIST 手写数字。
  - 训练参数：5 轮，批次大小 64，Adam 优化器 (lr=0.001)
  - 输出：每轮训练损失，最终测试准确率
  - 示例输出：
    ```
    开始训练...
    第 1 轮 损失: 0.234
    ...
    ✅ 最终准确率: 98.50%
    ✅ 模型已保存：mnist_cnn_best.pth
    ```

- **`lesson8.py`**:
  - 功能：加载训练好的 MNIST 模型，对本地图片进行预测。
  - 输入：本地图片路径 (默认 "test.png")
  - 预处理：灰度转换 + 调整到 28x28 + 反转处理
  - 输出：预测数字 + 置信度
  - 示例输出：
    ```
    ✅ 高精度模型加载成功！
    🎯 模型预测：7
    🎯 置信度：95.23%
    ```

### CIFAR10 彩色图像分类 (10 类)

- **`train.py`**:
  - 功能：训练 CNN 模型识别 CIFAR10 数据集。
  - 模型结构：2 层卷积 + 3 层全连接
  - 训练参数：10 轮，批次大小 64，Adam 优化器 (lr=0.001)
  - 输出：每轮平均损失
  - 示例输出：
    ```
    开始训练...
    第 1 轮，损失: 1.850
    ...
    ✅ 训练完成！模型已保存为 cifar10_model.pth
    ```

- **`lesson9.py`**:
  - 功能：加载训练好的 CIFAR10 模型，对本地彩色图片进行预测。
  - 输入：本地图片路径 (默认 "car.jpg")
  - 预处理：调整到 32x32 + 归一化
  - 输出：预测类别 (中文) + 置信度
  - 示例输出：
    ```
    🎯 预测：汽车
    🎯 置信度：87.45%
    ```

- **`camera.py`**:
  - 功能：实时摄像头 CIFAR10 分类，使用摄像头捕获帧并实时预测。
  - 输入：摄像头视频流
  - 输出：画面中显示预测类别 + 置信度 + 彩色框
  - 控制：按 'q' 退出
  - 特点：支持中文显示，无乱码

### 高级目标检测

- **`yolo.py`**:
  - 功能：使用 YOLOv8 Nano 进行实时摄像头目标检测。
  - 支持类别：80 类 COCO 数据集物体
  - 输入：摄像头视频流
  - 输出：实时标注画面 (检测框 + 类别 + 置信度)
  - 控制：按 'q' 退出

## 环境要求与安装

### 系统要求
- Python 3.8+
- CUDA 11.8+ (可选，用于 GPU 加速)
- 摄像头 (仅用于 yolo.py)

### 依赖包
- `torch` >= 2.0.0
- `torchvision` >= 0.15.0
- `ultralytics` >= 8.0.0 (仅 yolo.py)
- `opencv-python` >= 4.8.0 (仅 yolo.py)
- `Pillow` >= 10.0.0

### 安装步骤

1. **创建 conda 环境** (推荐)：
   ```bash
   conda create -n pytorch python=3.10
   conda activate pytorch
   ```

2. **安装 PyTorch** (根据你的 CUDA 版本选择)：
   ```bash
   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # CPU 版本
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **安装其他依赖**：
   ```bash
   pip install ultralytics opencv-python pillow
   ```

4. **验证安装**：
   ```python
   import torch
   print(torch.cuda.is_available())  # True 表示 GPU 可用
   ```

## 运行指南

### 1. 数据准备
- MNIST/CIFAR10 数据集会自动下载，无需手动准备。
- 本地图片预测：将图片放在项目根目录，命名为指定文件名。

### 2. 训练模型
```bash
# MNIST 训练 (约 5-10 分钟)
python lesson7.py

# CIFAR10 训练 (约 10-20 分钟，取决于硬件)
python train.py
```

### 3. 推理预测
```bash
# MNIST 本地图片 (准备 test.png)
python lesson8.py

# CIFAR10 本地图片 (准备 car.jpg 等)
python lesson9.py

# 实时目标检测
python yolo.py
```

### 4. 自定义运行
- 修改图片路径：在代码中更改 `image_path` 变量。
- 调整训练参数：在训练脚本中修改 `num_epochs`, `batch_size` 等。
- 更换 YOLO 模型：在 `yolo.py` 中将 `"yolov8n.pt"` 改为 `"yolov8s.pt"` 等。

## 数据集信息

- **MNIST**: 70,000 张 28x28 灰度手写数字图片
  - 训练集: 60,000 张
  - 测试集: 10,000 张
- **CIFAR10**: 60,000 张 32x32 彩色图片，10 类
  - 飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
  - 每类 6,000 张

## 模型文件说明

- `mnist_cnn_best.pth`: MNIST CNN 模型权重 (~1MB)
- `cifar10_model.pth`: CIFAR10 CNN 模型权重 (~3MB)
- YOLO 模型会自动下载到用户缓存目录

## 性能优化提示

- **GPU 加速**: 确保安装了 CUDA 版本的 PyTorch
- **批处理**: 训练时增大 `batch_size` 可加速 (需足够显存)
- **模型大小**: YOLO 使用 n/s/m/l/x 版本平衡速度和精度
- **图片质量**: 本地图片预测时，使用清晰、无噪点的图片

## 常见问题 (FAQ)

**Q: 运行时出现 CUDA 错误？**
A: 检查 CUDA 版本兼容性，或使用 CPU 版本：`torch.device('cpu')`

**Q: YOLO 下载模型慢？**
A: 使用国内镜像或手动下载模型文件到项目目录。

**Q: 本地图片预测不准确？**
A: 检查图片格式 (PNG/JPG)，确保是目标类别，尝试调整预处理参数。

**Q: 如何添加新类别？**
A: 需要重新训练模型，使用自定义数据集。

**Q: 训练时间太长？**
A: 减少训练轮数，或使用更简单的模型结构。

## 学习路径建议

1. **入门基础**: `lesson1.py` → 了解 PyTorch 环境和张量
2. **简单应用**: `lesson2.py` → 体验神经网络拟合
3. **数据与训练**: `lesson3.py` → MNIST 全连接训练
4. **模型管理**: `lesson4.py` → 学习保存/加载模型
5. **深度学习**: `lesson5.py` → CNN 卷积网络入门
6. **推理实践**: `lesson6.py` → 模型推理流程
7. **完整训练**: `lesson7.py` → CNN 训练 MNIST
8. **应用扩展**: `lesson8.py` → 本地图片预测
9. **彩色分类**: `train.py` + `lesson9.py` → CIFAR10 分类
10. **实时应用**: `camera.py` → 摄像头实时分类
11. **目标检测**: `yolo.py` → 高级目标检测

## 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [YOLOv8 文档](https://docs.ultralytics.com/)
- [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)
- [CIFAR10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)

## 许可证

本项目仅供学习使用，请遵守相关数据集和模型的许可证条款。

---

**作者**: AI Assistant  
**更新日期**: 2026-03-27  
**版本**: 1.0

有问题或建议欢迎反馈！🎉