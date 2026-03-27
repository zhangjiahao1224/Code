# Py\Projects\pytorch
# 导入PyTorch
import torch

# ====================== 1. 查看环境是否正常 ======================
print("===== 环境信息 =====")
print("PyTorch版本:", torch.__version__)
print("GPU是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("你的显卡:", torch.cuda.get_device_name(0))

# ====================== 2. 创建张量（Tensor） ======================
# 张量就是PyTorch里的数组，可以用GPU加速
print("\n===== 创建张量 =====")

# 1行3列的张量
x = torch.tensor([1.0, 2.0, 3.0])
print(x)

# 2行3列的随机数
y = torch.rand(2, 3)
print(y)

# ====================== 3. 把数据放到GPU上 ======================
print("\n===== 使用GPU =====")
if torch.cuda.is_available():
    x = x.cuda()  # 移动到GPU
    y = y.cuda()
    print("x所在设备:", x.device)  # 显示 cuda:0 就是成功

# ====================== 4. GPU上做计算 ======================
print("\n===== GPU计算 =====")
z = x + y
print(z)

# ====================== 5. 回到CPU ======================
z = z.cpu()
print("回到CPU:", z.device)