import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# 1. 定义一个简单的卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1_input_size = self._get_conv_output()  # 计算卷积层输出尺寸
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _get_conv_output(self):
        # 创建一个假的输入张量来模拟MNIST图像
        # 假设输入是28x28的单通道图像
        with torch.no_grad():
            # 使用模型的前半部分来获取最后一个卷积层的输出尺寸
            test_input = torch.zeros(1, 1, 28, 28)
            output = self.conv1(test_input)
            output = F.relu(output)
            output = self.conv2(output)
            output = F.relu(output)
            output = F.max_pool2d(output, 2)
            output = self.dropout1(output)
            # 计算输出的尺寸
            h, w = output.shape[2], output.shape[3]
            return output.numel()  # 输出的总元素数

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)  # 这里之前错误地使用了dropout1
        x = torch.flatten(x, 1)  # 展平除了批次维度的所有维度
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)  # 修正：使用dropout2而不是dropout1
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 定义图像预处理和预测函数
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # 'L' 模式代表灰度图像
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整图像大小到28x28
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批处理维度
    return image

def predict_image(model, image_path):
    image = load_image(image_path)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 实例化模型、损失函数和优化器
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')

# 保存模型权重
model_path = './model/mymodel'
file_path = Path(os.getcwd() + os.path.sep + model_path)
file_path.touch()
torch.save(model.state_dict(), file_path)  # 保存模型权重