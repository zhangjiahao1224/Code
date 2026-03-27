# Py\Projects\pytorch
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import PIL.ImageOps

# 设备
device = torch.device("cuda")
print("使用设备:", device)

# CNN模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn_best.pth", weights_only=True))
model.eval()

# 最简单、最稳定的预处理（不改你图片位置大小）
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# 只反色 + 灰度，不移动、不拉伸、不居中
img = Image.open("test.png").convert("L")
img = PIL.ImageOps.invert(img)
img = transform(img).unsqueeze(0).to(device)

# 预测
with torch.no_grad():
    output = model(img)
    prob = torch.softmax(output, dim=1)
    pred = output.argmax().item()
    conf = prob[0][pred].item() * 100

print(f"\n🎯 你画的数字：{pred}")
print(f"🎯 置信度：{conf:.2f}%")