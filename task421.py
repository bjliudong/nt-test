import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Dataset

# 定义一个简单的数据集类，用于加载单个图像
class SingleImageDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path
        # 定义转换操作，将图像转换为模型可接受的格式
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # 转换为灰度图
            transforms.Resize((28, 28)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5,), (0.5,)),  # 标准化
        ])
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert('L')  # 'L' 模式表示灰度图
        return self.transform(image)

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=None)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义一个简单的卷积神经网络模型
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleCNN()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(5):  # 简单示例，只训练5个epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/5], Loss: {loss.item():.4f}')

# 加载待识别的图像
image_path = 'digit.png'  # 替换为您的图像路径
dataset = SingleImageDataset(image_path)
dataloader = DataLoader(dataset, batch_size=1)

# 识别图像
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    for images, _ in dataloader:
        output = model(images)
        _, predicted = torch.max(output, 1)
        print(f"Recognized digit: {predicted.item()}")

# 注意：这里的图像路径 'digit.png' 需要替换为您实际的图像路径