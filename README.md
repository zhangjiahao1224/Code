# 机器人与 AI 代码仓库

这个仓库用于整理机器人、机器学习、深度学习、计算机视觉，以及通用编程练习相关的代码与资料。内容既包含学习阶段的示例和实验，也包含课程实践、小项目、工具脚本，以及运行后生成的数据与产物。

目前仓库以 `Python / C++ / Matlab` 三条主线组织，适合作为长期积累式的学习仓库来使用。

## 仓库定位

- 按语言划分主要代码区域：`Py/`、`C++/`、`Matlab/`
- 按用途区分学习代码、项目实践、工具脚本、数据和实验产物
- 保留历史实验与归档内容，方便回看和复用

## 目录导航

### 顶层目录

| 目录 | 说明 |
| --- | --- |
| `Py/` | Python 相关代码，包含学习、项目、脚本、ROS、工具、刷题与数据分析 |
| `C++/` | C++ 学习、算法、ROS、项目与工具库 |
| `Matlab/` | Matlab 机器人仿真、轨迹与数据分析相关代码 |
| `Common/` | 共享资源、配置或通用内容 |
| `data/` | 当前使用中的数据资源 |
| `artifacts/` | 训练结果、导出图像、模型权重等运行产物 |
| `Archive/` | 历史项目、旧实验与归档代码 |
| `zip/` | 压缩包、离线文件或安装包 |

### Python 重点入口

- [`Py/Projects/DeepLearning/`](./Py/Projects/DeepLearning/)：一组按课程推进的深度学习实践脚本，覆盖 MLP、序列建模、目标检测、生成模型与强化学习
- [`Py/Projects/DeepLearning/README.md`](./Py/Projects/DeepLearning/README.md)：DeepLearning 课程实践总览
- [`Py/Projects/DeepLearning/Class_5.py`](./Py/Projects/DeepLearning/Class_5.py)：第 5 课强化学习示例，包含 DQN 与 Policy Gradient
- [`Py/Projects/pytorch/`](./Py/Projects/pytorch/)：PyTorch 入门练习、MNIST/CIFAR10 训练与 YOLO 实验
- [`Py/Projects/MachineLearning/`](./Py/Projects/MachineLearning/)：机器学习课程资料与网页资源
- [`Py/Projects/snake_bot/`](./Py/Projects/snake_bot/)：基于屏幕识别与路径规划的贪吃蛇机器人
- [`Py/Learning/`](./Py/Learning/)：Python 基础与练习代码
- [`Py/LeetCode/`](./Py/LeetCode/)：算法刷题记录 
- [`Py/Scripts/`](./Py/Scripts/)：课程脚本、Notebook 与辅助实验材料
- [`Py/Utils/`](./Py/Utils/)：Python 通用工具函数

### C++ 重点入口

- [`C++/Learning/`](./C++/Learning/)：C++ 基础、STL、内存管理与现代 C++ 练习
- [`C++/Algorithm/`](./C++/Algorithm/)：算法与实现练习
- [`C++/ROS/`](./C++/ROS/)：ROS 相关代码
- [`C++/Projects/`](./C++/Projects/)：C++ 项目实践
- [`C++/Libs/`](./C++/Libs/)：自定义或整理后的工具库

### Matlab 重点入口

- [`Matlab/Robotics/`](./Matlab/Robotics/)：机器人仿真、轨迹规划与控制实验
- [`Matlab/DataAnalysis/`](./Matlab/DataAnalysis/)：Matlab 数据分析与可视化

## 当前值得优先查看的内容

如果你是从根目录开始看，下面几个入口最能代表目前仓库的主要实践方向：

1. [`Py/Projects/DeepLearning/README.md`](./Py/Projects/DeepLearning/README.md)
2. [`Py/Projects/DeepLearning/Class_1.py`](./Py/Projects/DeepLearning/Class_1.py)
3. [`Py/Projects/DeepLearning/Class_4.py`](./Py/Projects/DeepLearning/Class_4.py)
4. [`Py/Projects/DeepLearning/Class_5.py`](./Py/Projects/DeepLearning/Class_5.py)
5. [`Py/Projects/pytorch/`](./Py/Projects/pytorch/)

其中：

- `Class_1.py` 适合理解监督学习训练闭环
- `Class_4.py` 适合理解 Autoencoder / VAE / GAN
- `Class_5.py` 适合理解强化学习中的 DQN 与策略梯度

## 建议的阅读顺序

### 如果你想从基础开始

1. 先看 `Py/Learning/` 和 `C++/Learning/`
2. 再看 `Py/Projects/pytorch/`，熟悉 PyTorch 基础训练与推理流程
3. 最后进入 `Py/Projects/DeepLearning/`，按 `Class_1 -> Class_5` 顺序阅读

### 如果你想直接看深度学习实践

1. 从 [`Py/Projects/DeepLearning/README.md`](./Py/Projects/DeepLearning/README.md) 开始
2. 按课程顺序阅读 `Class_1.py` 到 `Class_5.py`
3. 再结合 `Py/Projects/DeepLearning/artifacts/` 中的结果文件理解训练输出

### 如果你想找历史资料或旧实验

- 进入 `Archive/`
- 进入 `Py/Scripts/`
- 进入 `Py/Projects/d2l-zh-master/` 或 `Py/Projects/tutorials-main/`

## 简化结构示意

```text
.
├── C++/
│   ├── Algorithm/
│   ├── Learning/
│   ├── Libs/
│   ├── Projects/
│   └── ROS/
├── Py/
│   ├── DataAnalysis/
│   ├── Learning/
│   ├── LeetCode/
│   ├── LeRobot/
│   ├── Libs/
│   ├── Projects/
│   ├── ROS/
│   ├── Scripts/
│   ├── Test/
│   └── Utils/
├── Matlab/
│   ├── DataAnalysis/
│   └── Robotics/
├── Common/
├── data/
├── artifacts/
├── Archive/
└── zip/
```

## 环境建议

- Python：建议 `Python 3.10+`
- C++：建议 `C++17+`
- Matlab：建议 `R2020b+`
- 深度学习相关脚本建议使用独立虚拟环境，并按项目分别安装依赖

## 使用说明

- 仓库里部分目录会保存训练权重、图像输出、日志等实验产物
- 一些数据集和模型文件体积较大，可能已经存在于仓库中，也可能由脚本首次运行时自动生成
- 如果某个子目录下已有自己的 `README.md`，优先以子目录文档为准

## 备注

- 根目录 `README.md` 负责提供仓库总览与导航
- 更具体的实验说明请查看对应子项目文档
- 当前深度学习课程实践的核心说明位于 [`Py/Projects/DeepLearning/README.md`](./Py/Projects/DeepLearning/README.md)
