# 机器人与 AI 代码仓库

这个仓库用于整理机器人、深度学习、视觉、数据分析以及通用编程练习代码。内容既包括学习阶段的示例与笔记，也包括课程实践、小型项目和历史实验归档。

## 仓库定位

- 按语言组织：`C++`、`Py`、`Matlab`
- 按用途拆分：学习示例、项目实践、工具库、数据与产物
- 适合用来持续积累机器人与 AI 方向的代码资产

## 目录导航

### 顶层目录

| 目录 | 说明 |
| --- | --- |
| `C++/` | C++ 相关代码，包含 ROS、算法、工具库与学习示例 |
| `Py/` | Python 相关代码，包含项目、深度学习练习、脚本、数据分析与刷题 |
| `Matlab/` | Matlab 相关代码，主要用于机器人仿真与数据分析 |
| `Common/` | 跨语言共享资源，如配置、数据集与文档 |
| `data/` | 当前使用中的数据资源 |
| `artifacts/` | 训练结果、中间产物或导出文件 |
| `Archive/` | 历史实验、旧项目与归档代码 |
| `zip/` | 压缩包、安装包等离线资源 |

### 重点入口

#### Python

- [`Py/Projects/DeepLearning/`](./Py/Projects/DeepLearning/)：深度学习课程实践，涵盖 `MLP`、`RNN`、`Self-Attention` 与 `Transformer`
- [`Py/Projects/DeepLearning/README.md`](./Py/Projects/DeepLearning/README.md)：深度学习课程实践说明
- [`Py/Projects/MachineLearning/`](./Py/Projects/MachineLearning/)：机器学习相关实验
- [`Py/Projects/pytorch/`](./Py/Projects/pytorch/)：PyTorch 练习与数据集实验
- [`Py/Learning/`](./Py/Learning/)：Python 基础与进阶学习代码
- [`Py/LeetCode/`](./Py/LeetCode/)：算法刷题记录
- [`Py/LeRobot/`](./Py/LeRobot/)：LeRobot 相关数据、训练与部署流程

#### C++

- [`C++/ROS/`](./C++/ROS/)：ROS 节点、实时控制与机器人相关程序
- [`C++/Algorithm/`](./C++/Algorithm/)：路径规划、SLAM、控制算法实现
- [`C++/Libs/`](./C++/Libs/)：自封装 C++ 工具库
- [`C++/Learning/`](./C++/Learning/)：C++ 基础、STL、内存管理与现代 C++ 练习

#### Matlab

- [`Matlab/Robotics/`](./Matlab/Robotics/)：机器人仿真与控制算法验证
- [`Matlab/DataAnalysis/`](./Matlab/DataAnalysis/)：数据分析与可视化脚本

## 简化结构示意

```text
.
├── C++/
│   ├── ROS/
│   ├── Algorithm/
│   ├── Libs/
│   ├── Projects/
│   └── Learning/
├── Py/
│   ├── Learning/
│   ├── Projects/
│   ├── LeRobot/
│   ├── DataAnalysis/
│   ├── Scripts/
│   ├── ROS/
│   ├── Utils/
│   └── LeetCode/
├── Matlab/
│   ├── Robotics/
│   └── DataAnalysis/
├── Common/
│   ├── Configs/
│   ├── Datasets/
│   └── Docs/
├── data/
├── artifacts/
├── Archive/
└── zip/
```

## 使用建议

- 想看基础学习内容时，优先从 `Py/Learning/` 和 `C++/Learning/` 开始
- 想看课程实践与模型实现时，可直接进入 `Py/Projects/DeepLearning/`
- 想找历史实验或旧项目，可在 `Archive/` 中查找
- 公共配置、文档和数据集统一放在 `Common/` 下，方便跨项目复用

## 环境要求

- C++：`CMake`、`C++17+`、部分项目依赖 `ROS`
- Python：建议 `Python 3.8+`
- Matlab：建议 `R2020b+`
