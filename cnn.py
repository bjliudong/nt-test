import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 2. 加载数据集
trainset = datasets.MNIST('./data/', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('./data/', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 3. 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道1, 输出通道32, 3x3卷积核
        self.pool1 = nn.MaxPool2d(2, 2)  # 池化层, 2x2池化核, 步长为2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 第二个卷积层
        self.pool2 = nn.MaxPool2d(2, 2)  # 池化层, 2x2池化核, 步长为2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层, 根据卷积层输出调整维度
        self.fc2 = nn.Linear(128, 10)  # 输出层, 10个输出对应10个类别

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # 卷积后接激活函数和池化
        x = self.pool2(F.relu(self.conv2(x)))  # 第二个卷积层后同样接池化
        x = x.view(x.size(0), -1)  # Flatten the tensor to feed into fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = CNN()

# 4. 选择损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. 训练模型
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images = images.view(images.size(0), 1, 28, 28)  # 转换为卷积层需要的尺寸
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')

# 6. 测试模型性能
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.view(images.size(0), 1, 28, 28)  # 转换为卷积层需要的尺寸
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

accuracy = test(model, testloader)
print(f'Test Accuracy: {accuracy:.2f}%')