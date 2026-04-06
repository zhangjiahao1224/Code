import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("评估设备:", device)

# 类别
classes = ["飞机","汽车","鸟","猫","鹿","狗","青蛙","马","船","卡车"]

# 模型（和你第9课一模一样）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 加载你之前训练好的模型
model = Net().to(device)
model.load_state_dict(torch.load("cifar10_model.pth", weights_only=True))
model.eval()

# 测试集
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 统计
correct = 0
total = 0
confusion = np.zeros((10,10), dtype=int)

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 混淆矩阵
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion[t.item(), p.item()] += 1

# 输出总分
acc = 100 * correct / total
print(f"\n✅ 整体准确率: {acc:.2f} %")

# 每一类准确率
print("\n📊 每一类单独得分：")
for i in range(10):
    class_acc = confusion[i,i] / confusion[i].sum() * 100
    print(f"{classes[i]:<4} : {class_acc:.2f} %")

# 画混淆矩阵
plt.figure(figsize=(10,8))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.title("CIFAR10 混淆矩阵")
plt.show()