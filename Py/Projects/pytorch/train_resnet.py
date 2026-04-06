import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------
# 4090 专属：满血 CUDA
# -------------------
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True  # 4090 加速
torch.cuda.empty_cache()

print("✅ 训练设备：GPU")

# -------------------
# 标准 ResNet 预处理
# -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------
# CIFAR10
# -------------------
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

# -------------------
# ResNet18 预训练模型
# -------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 冻结主干
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# -------------------
# 训练配置
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# -------------------
# 开始训练
# -------------------
print("\n🔥训练开始！")
model.train()

for epoch in range(10): # 10 轮
    loss_total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    print(f"Epoch {epoch+1} | Loss: {loss_total/len(train_loader):.3f}")

torch.save(model.state_dict(), "resnet_cifar10.pth")
print("\n🎉 训练完成！模型：resnet_cifar10.pth")
