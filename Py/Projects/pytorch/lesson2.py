# Py\Projects\pytorch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 用 GPU，如果没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ----------------------
# 1. 造一批假数据
# ----------------------
x = torch.linspace(0, 10, 100).view(-1, 1).to(device)  # x 从0到10
y = 2 * x + 5 * torch.randn_like(x)  # y = 2x + 噪音

# ----------------------
# 2. 搭建超级简单的神经网络
# ----------------------
model = nn.Linear(1, 1).to(device)

# 优化器 + 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = nn.MSELoss()

# ----------------------
# 3. 开始训练（AI学习）
# ----------------------
print("\n开始训练...")
for epoch in range(500):
    pred = model(x)
    loss = loss_func(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"训练轮次 {epoch:3d} | 损失: {loss.item():.4f}")

# ----------------------
# 4. 训练完成，看结果
# ----------------------
print("\n训练完成！")
w = model.weight.item()
b = model.bias.item()
print(f"AI 学到的公式：y = {w:.2f}x + {b:.2f}")
print("本来的公式：y = 2x + 0")