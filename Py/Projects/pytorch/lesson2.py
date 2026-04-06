# Py\Projects\pytorch 示例：线性回归基础（从0开始训练一个单层模型）
# 包含：数据构造、模型定义、训练循环、结果验证

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 选择计算设备
# 优先使用 GPU（cuda），否则回退到 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ----------------------
# 2. 构造模拟数据
# ----------------------
# x: 从 0 到 10，100 个点，形状 [100, 1]
x = torch.linspace(0, 10, 100).view(-1, 1).to(device)
# y: 真实函数 y=2x，加入随机噪声以模拟真实数据
# 这里噪声幅度为 5 × 正态分布，偏离真实直线
y = 2 * x + 5 * torch.randn_like(x)

# ----------------------
# 3. 模型定义与优化器
# ----------------------
# 使用最简单的线性模型：y = wx + b
model = nn.Linear(1, 1).to(device)

# Adam 优化器，学习率 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# 均方误差损失（回归问题常用）
loss_func = nn.MSELoss()

# ----------------------
# 4. 训练循环
# ----------------------
print("\n开始训练...")
# 迭代 500 次尝试拟合参数
for epoch in range(500):
    # 前向计算输出
    pred = model(x)
    # 计算损失（预测值与真实值之间误差）
    loss = loss_func(pred, y)

    # 反向传播步骤
    optimizer.zero_grad()   # 清零梯度缓存
    loss.backward()         # 自动求梯度
    optimizer.step()        # 更新参数

    # 每 50 轮打印一次损失
    if epoch % 50 == 0:
        print(f"训练轮次 {epoch:3d} | 损失: {loss.item():.4f}")

# ----------------------
# 5. 输出训练结果
# ----------------------
print("\n训练完成！")
# 从模型参数中读取学习到的 w 和 b
w = model.weight.item()
b = model.bias.item()
print(f"AI 学到的公式：y = {w:.2f}x + {b:.2f}")
print("本来的公式：y = 2x + 0")

# 可选：绘图展示拟合结果
# x_np = x.cpu().detach().numpy()
# y_np = y.cpu().detach().numpy()
# y_pred_np = model(x).cpu().detach().numpy()
# plt.scatter(x_np, y_np, label='真实数据')
# plt.plot(x_np, y_pred_np, 'r-', label='拟合直线')
# plt.legend()
# plt.show()