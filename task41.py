import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# 定义数据集类
class DrugDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 实例化数据集
dataset = DrugDataset(images=['path_to_images'], labels=['path_to_labels'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义CRNN模型
class CRNN(torch.nn.Module):
    # 定义模型结构
    def __init__(self):
        super(CRNN, self).__init__()
        # 添加模型层

    def forward(self, x):
        # 定义前向传播过程
        return x

# 实例化模型
model = CRNN()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
for epoch in range(num_epochs):
    for images, labels in dataloader:
        # 将图片和标签转换为tensor
        images = torch.stack(images).view(-1, 1, 32, 128)
        labels = torch.tensor(labels)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 模型评估和部署
# ...

# 使用模型进行预测
# ...