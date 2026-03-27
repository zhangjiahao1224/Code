## PyTorch 学习路线（基于本仓库）

### 1. 学习目标
- **理解基础张量操作**：会用 `torch.Tensor` 做加减乘除、形状变换等。
- **掌握常见模块**：`nn.Module`、`DataLoader`、优化器、损失函数。
- **能独立完成一个小项目**：如 MNIST 手写数字分类或简单图像分类。

---

### 2. 环境准备
1. 安装 Python（建议 3.9+）。
2. 按照官方指引安装 PyTorch：参见 [`https://pytorch.org/get-started`](https://pytorch.org/get-started)。
3. 在本仓库根目录执行：
   - `pip install -r requirements.txt`
4. 如果你要本地构建文档或运行全部教程，建议使用 GPU 环境；仅阅读/运行部分教程则 CPU 也可以。

---

### 3. 推荐学习顺序（初学者）
1. **官网 Beginner 教程（优先）**
   - 打开：[`https://pytorch.org/tutorials`](https://pytorch.org/tutorials)
   - 在左侧目录中找到 `Beginner`，依次学习：
     - `What is PyTorch`
     - `Tensors`
     - `Autograd`
     - `Neural Network` / `Training a classifier`
2. **本仓库中对应源码**
   - 查看 `beginner_source/` 目录下的 `.py` 文件，一般和官网 Beginner 教程一一对应。
   - 建议步骤：
     1. 先在网页上通读教程。
     2. 再在本地打开对应 `.py`，逐行理解并加上自己的注释。
     3. 修改一些超参数（如学习率、批大小）观察训练效果变化。

---

### 4. 如何使用本仓库的教程
1. **只想看教程代码 / 跑一下示例**
   - 直接在 `beginner_source/`、`intermediate_source/` 中挑选感兴趣的 `.py` 或 `.ipynb`（若存在）。
   - 在本地用 `python your_tutorial.py` 运行，或复制核心代码到自己的项目中练习。
2. **想查看生成的 HTML 文档**
   - 机器有 GPU 时：在仓库根目录执行 `make docs`，生成的文档在 `docs/` 目录。
   - 机器无 GPU 时：执行 `make html-noplot`，生成基础 HTML 文档在 `_build/html/`。
3. **只构建某一个教程**
   - 在命令行中执行（示例）：
     ```bash
     GALLERY_PATTERN="neural_style_transfer_tutorial.py" make html
     ```

---

### 5. 每周学习节奏建议（参考）
- **第 1 周**：完成张量基础和自动求导相关教程；手写几个小例子（线性回归、简单函数拟合）。
- **第 2 周**：完成基础神经网络和分类任务教程；训练一个小型分类模型（如 MNIST）。
- **第 3 周及以后**：根据兴趣选择 `intermediate_source/` 中的教程（如 RNN、Transformer、计算机视觉、NLP 等）。

---

### 6. 练习建议
- 每看完一个教程，尝试：
  - 改一个数据集（如从 MNIST 换成自己准备的小数据）。
  - 改一个网络结构（如增加一层、改激活函数）。
  - 画出 loss 曲线，写一段简短总结：**本次实验我改变了什么，结果如何，可能的原因是什么**。

---

### 7. 下一步可以做什么
1. 在浏览器打开：[`https://pytorch.org/tutorials`](https://pytorch.org/tutorials)。
2. 在仓库中找到 `beginner_source` 目录，对照官网教程，从第一个 Beginner 教程开始照着敲代码。
3. 遇到看不懂的地方，可以把对应代码片段贴给我，我帮你中文讲解和补充注释。

