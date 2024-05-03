accuracies = []

# 对每个采样比例训练模型并在测试集上评估
for percentage in percentages:
    sampled_train_dataset = sample_dataset(train_dataset, percentage)
    subset_loader = torch.utils.data.DataLoader(
        sampled_train_dataset, batch_size=64, shuffle=True)

    # 训练模型
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, epochs + 1):
        for data, target in subset_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    # 在完整的测试集上评估模型性能
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    accuracies.append(accuracy)
    print(f'Accuracy for {percentage*100}% of training set: {accuracy:.2f}%')

# 绘制准确率图表
plt.figure(figsize=(8, 6))
plt.plot(percentages, accuracies, marker='o')
plt.xlabel('Training Set Percentage')
plt.ylabel('Accuracy on Test Set (%)')
plt.title('Model Accuracy vs Training Set Size')
plt.grid(True)
plt.show()