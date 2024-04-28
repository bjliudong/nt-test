# 作业要求

建立神经网络模型，隐含层数量不超过3层，其中至少有1层为全连接层。给出网络结构的所有参数设置情况，包括网络层数、每层神经元数量、活化函数的选择、学习率的设定、损失函数的定义等。对网络的输入进行描述，即原始特征或其他特征的表示方式。

# 参考答案

`mlp.py` 是基于Pytorch的MLP实现代码。

## 网络结构参数设置

1. **网络层数**：在示例中，我们定义了一个包含三个全连接层的MLP网络。
  - 第一层（输入层）：不显式定义，因为输入数据直接通过view操作展平。
  - 第二层：具有512个神经元的全连接层。
  - 第三层：具有128个神经元的全连接层。
  - 第四层（输出层）：具有10个神经元的全连接层，对应10个数字类别。

2. **每层神经元数量**：
  - 第二层：512个神经元。
  - 第三层：128个神经元。
  - 第四层：10个神经元。

3. **激活函数的选择**：在MLP中，我们使用了ReLU（Rectified Linear Unit）激活函数，它在正区间内是线性的，负区间内输出为零。

4. **学习率的设定**：在优化器optim.Adam中，我们设置了学习率为0.001。

5. **损失函数的定义**：我们使用了交叉熵损失函数F.cross_entropy，它适用于多分类问题。

## 网络输入描述

1. **原始特征**：MNIST数据集由28x28像素的手写数字图像组成，每个像素的值在0到1之间（通过归一化处理）。

2. **输入特征表示方式**：输入图像首先通过transforms模块进行转换，将PIL图像转换为PyTorch张量，然后使用transforms.Normalize进行归一化处理，最后通过view操作将图像展平为一维向量，以作为MLP的输入。

3. **数据预处理**：在加载数据集时，我们应用了transforms.Compose，其中包括将图像转换为张量和归一化操作。归一化是通过对图像张量减去均值（0.1307）并除以标准差（0.3081）来完成的。

## 实验内容

实验的具体内容可能包括：

- **数据集划分**：将MNIST数据集分为训练集和测试集。
- **模型初始化**：根据上述网络结构初始化MLP模型。
- **训练过程**：使用定义的训练函数对模型进行训练，包括前向传播、计算损失、反向传播和参数更新。
- **性能评估**：在每个训练周期结束后，使用测试函数评估模型在测试集上的性能，包括损失和准确率。
- **超参数调整**：可能需要调整的超参数包括学习率、批量大小、网络层数和神经元数量、激活函数以及优化器类型等。
- **结果分析**：对训练过程中的损失和准确率进行分析，确定模型的泛化能力和是否存在过拟合或欠拟合现象。

在实验过程中，可能需要多次调整上述参数，以获得最佳的模型性能。此外，实验报告中还应包括模型训练的具体细节、实验结果的可视化（如损失曲线、准确率曲线等）以及对结果的讨论。