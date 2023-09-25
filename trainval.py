"""
网络：
    resnet18
数据集：
    cifar-10
训练：
    ——learning rate warmup
    ——low-precision training
优化方法：
    ——model architecture tweaks
    ——cosine learning rate decay
    ——label smoothing
    knowledge distillation
    ——data augmentation
任务：
    classification
    detection
    segmentation
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import pandas as pd
from math import pi, cos
import numpy as np
import my_model


# 学习率预热
def warmup_lr(optimizer, epoch, init_lr, warmup_epochs):
    if epoch < warmup_epochs:
        lr = init_lr * (epoch + 1) / warmup_epochs
        lrs.append(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# 学习率衰减策略
def adjust_learning_rate(optimizer, epoch, initial_lr):
    # 线性学习率衰减
    # if epoch >= 4:
    #     lr = initial_lr * (1 - (epoch - 4) / num_epochs)
    #     lrs.append(lr)
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    # 余弦学习率衰减
    if epoch >= 4:
        lr = 0.5 * initial_lr * (1 + cos(pi * (epoch - 4) / num_epochs))
        lrs.append(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# 平滑标签
def smooth_labels(one_hot_labels, label_smoothing):
    num_classes = one_hot_labels.size(1)
    return one_hot_labels * (1.0 - label_smoothing) + label_smoothing / num_classes


# 画图和验证
def post_process():
    plt.figure(figsize=(12, 5))
    # 学习率
    plt.subplot(1, 3, 1)
    plt.plot(lrs, label='lr')
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    plt.title('lr Over Epochs')
    plt.legend()
    plt.grid(True)

    # 训练和测试损失曲线
    plt.subplot(1, 3, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)

    # 训练和测试精度曲线
    plt.subplot(1, 3, 3)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("result.png")
    # plt.show()

    # 创建一个包含数据的字典
    data_dict = {
        'Epoch': list(range(1, num_epochs + 1)),
        'Learning Rate': lrs,
        'Train Loss': train_losses,
        'Test Loss': test_losses,
        'Train Accuracy (%)': train_acc,
        'Test Accuracy (%)': test_acc
    }

    # 创建DataFrame对象
    df = pd.DataFrame(data_dict)

    # 将DataFrame保存为Excel文件
    excel_file_path = "training_results.xlsx"
    df.to_excel(excel_file_path, index=False)  # 保存为Excel文件，不包括索引列

    print(f"Data saved to {excel_file_path}")

    net.eval()  # 将模型设置为评估模式
    running_loss = 0
    correct = 0
    total = 0
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    start_time = time.time()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # images, labels = images.to(device), labels.to(device)
            images, labels = images.to(device, dtype=torch.float16), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    test_losses.append(epoch_loss)
    epoch_accuracy = 100 * correct / total
    test_acc.append(epoch_accuracy)
    end_time = time.time()
    test_time = end_time - start_time
    print(
        f"Test Loss: {epoch_loss:.4f},  Test Accuracy: {epoch_accuracy:.2f}%,  Test Time: {test_time:.2f}s")


if __name__ == "__main__":

    torch.manual_seed(17)  # 设置随机种子以保持实验结果的一致性
    label_smoothing = 0.1
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_pro = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机颜色调整
        transforms.RandomGrayscale(p=0.1),  # 随机灰度化
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载cafir-10数据集
    train_set = torchvision.datasets.CIFAR10(root='D:/myproject/datasets', train=True, download=False, transform=transform_pro)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
    test_set = torchvision.datasets.CIFAR10(root='D:/myproject/datasets', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    # 定义ResNet-18模型
    net = my_model.ResNet18()
    net = net.to(dtype=torch.float16)  # 低精度训练

    # 修改输出层以适应CIFAR-10数据集的10个类别
    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 10)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    lrs = []
    init_lr = 0.1
    optimizer = optim.SGD(net.parameters(), lr=init_lr, momentum=0, weight_decay=0)

    # 训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    num_epochs = 60
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(num_epochs):
        net.train()  # 将模型设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        warmup_lr(optimizer, epoch, init_lr, 5)
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.to(device, dtype=torch.float16), labels.to(device)
            smoothed_labels = smooth_labels(torch.nn.functional.one_hot(labels, num_classes=10).float(), label_smoothing)  # 平滑标签

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, smoothed_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算并打印每个epoch训练集的平均损失值和准确率
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        epoch_accuracy = 100 * correct / total
        train_acc.append(epoch_accuracy)

        adjust_learning_rate(optimizer, epoch, init_lr)
        end_time = time.time()
        train_time = end_time - start_time
        print(
            f"Epoch {epoch + 1} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, Train Time: {train_time:.2f}s")

        # 在测试集上评估模型
        net.eval()  # 将模型设置为评估模式
        running_loss = 0
        correct = 0
        total = 0

        start_time = time.time()
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                # images, labels = images.to(device), labels.to(device)
                images, labels = images.to(device, dtype=torch.float16), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(test_loader)
        test_losses.append(epoch_loss)
        epoch_accuracy = 100 * correct / total
        test_acc.append(epoch_accuracy)
        end_time = time.time()
        test_time = end_time - start_time
        print(
            f"          Test Loss: {epoch_loss:.4f},  Test Accuracy: {epoch_accuracy:.2f}%,  Test Time: {test_time:.2f}s")

    lrs.pop()

    # 绘制损失和精度曲线图
    post_process()
