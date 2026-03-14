机器人相关代码仓库

## 目录结构

```
├── C++/                  # C++ 相关代码
│   ├── ROS/              # ROS 节点、实时控制、运动学解算
│   ├── Algorithm/        # 路径规划、SLAM、控制算法实现
│   ├── Libs/             # 自己封装的C++工具库
│   ├── Projects/         # 完整C++项目
│   ├── Learning/         # C++学习路径
│   │   ├── 01-Basics/            # 变量、数据类型、运算符、流程控制
│   │   │   └── basics_demo.cpp
│   │   ├── 02-Functions/         # 函数定义、参数、返回值、重载
│   │   │   └── functions_demo.cpp
│   │   ├── 03-Arrays-and-Pointers/ # 数组、指针、引用
│   │   │   └── arrays_pointers_demo.cpp
│   │   ├── 04-OOP/               # 类、封装、继承、多态
│   │   │   └── oop_demo.cpp
│   │   ├── 05-STL/               # 标准库：容器、算法、迭代器
│   │   │   └── stl_demo.cpp
│   │   ├── 06-Memory-Management/ # new/delete、RAII、智能指针
│   │   │   └── memory_demo.cpp
│   │   └── 07-Advanced-Topics/   # 模板、异常、多线程、现代 C++ 特性
│   │       └── advanced_demo.cpp
│   ├── test.cpp          # 测试代码
│   └── test.exe          # 编译产物
│
├── Py/                   # Python 相关代码
│   ├── LeRobot/          # LeRobot 全流程
│   │   ├── data/          # 采集的 episode 数据
│   │   ├── train/         # 训练脚本
│   │   └── deploy/        # 推理部署
│   ├── ROS/              # Python ROS节点
│   ├── DataAnalysis/     # 数据分析、可视化
│   ├── Scripts/          # 自动化脚本
│   ├── Libs/             # 自己封装的Python库
│   ├── Utils/            # 常用工具函数
│   ├── Projects/         # Python 项目
│   ├── Learning/         # Python学习路径
│   │   ├── 01-Basics/           # 变量、数据类型、运算符、流程控制
│   │   │   └── basics_demo.py
│   │   ├── 02-Functions/        # 函数定义、参数、返回值、递归、lambda
│   │   │   └── functions_demo.py
│   │   ├── 03-Data-Structures/  # 列表、元组、字典、集合、字符串
│   │   │   └── data_structures_demo.py
│   │   ├── 04-OOP/              # 类、封装、继承、多态、特殊方法
│   │   │   └── oop_demo.py
│   │   ├── 05-File-Handling/    # 文件读写、CSV处理、JSON处理、路径操作
│   │   │   └── file_handling_demo.py
│   │   ├── 06-Modules-and-Packages/  # 模块导入、包结构、标准库、虚拟环境
│   │   │   └── modules_packages_demo.py
│   │   ├── 07-Exception-Handling/    # 异常处理、自定义异常、断言、日志记录
│   │   │   └── exception_handling_demo.py
│   │   └── 08-Standard-Library/  # 常用标准库：os, sys, datetime, random, math, itertools
│   │       └── standard_library_demo.py
│   ├── 1Learning/        # 学习资料
│   └── Test/             # 测试代码
│
├── Matlab/               # Matlab 相关代码
│   ├── Robotics/         # 机器人仿真、控制算法验证
│   ├── DataAnalysis/     # 数据分析、可视化
│   └── RTB.mltbx         # Robotics Toolbox 工具箱
│
├── Common/               # 跨语言共享资源
│   ├── Configs/          # 机器人配置文件、URDF
│   ├── Datasets/         # 机械臂数据集
│   └── Docs/             # 项目文档
│
├── Archive/              # 旧代码、实验归档
├── zip/                  # 压缩包、安装包
└── download.png          # 下载资源
```

## 环境要求

- C++: ROS, CMake, C++17+
- Python: 3.8+
- Matlab: R2020b+
