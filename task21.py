import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 3. 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 512)  # 28x28 -> 512
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)   # 10类输出

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 原始训练集
full_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)

# 定义采样函数
def sample_dataset(dataset, percentage):
    num_samples = int(len(dataset) * percentage)
    indices = torch.randperm(len(dataset))[:num_samples]
    return torch.utils.data.Subset(dataset, indices)

# 定义训练和评估过程
def train_and_evaluate(subset_loader, valid_loader=None, epochs=5):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []
    accuracies = []

    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        correct = 0
        total = 0

        # 训练
        model.train()
        for data, target in subset_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_losses.append(train_loss / len(subset_loader))
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        # 验证
        if valid_loader:
            model.eval()
            valid_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    valid_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            valid_losses.append(valid_loss / len(valid_loader))
            print(f'Validation set: Average loss: {valid_loss / len(valid_loader):.4f}, Accuracy: {100 * correct / total:.0f}%')

    return train_losses, valid_losses, accuracies

# 采样比例
percentages = [0.001, 0.01, 0.1]

# 绘制图表
plt.figure(figsize=(10, 8))

for i, percentage in enumerate(percentages):
    sampled_train_dataset = sample_dataset(train_dataset, percentage)
    if percentage > 0.1:  # 抽取10%以上时，创建验证集
        valid_dataset = sample_dataset(sampled_train_dataset, 0.1)
        subset_loader = torch.utils.data.DataLoader(
            sampled_train_dataset, batch_size=64, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=64, shuffle=False)
    else:
        subset_loader = torch.utils.data.DataLoader(
            sampled_train_dataset, batch_size=64, shuffle=True)
        valid_loader = None

    train_losses, valid_losses, accuracies = train_and_evaluate(
        subset_loader, valid_loader, epochs=5)
    
    plt.subplot(2, 1, i + 1)
    plt.plot(train_losses, label='Train Loss')
    if valid_loader:
        plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for {percentage*100}% of Dataset')
    plt.legend()

plt.tight_layout()
plt.show()