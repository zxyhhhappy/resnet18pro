import torch  # 开源的深度学习框架，提供张量计算、自动微分和深度学习模型，张量tensor可以GPU加速
import torch.nn as nn  # nn=neural networks，提供网络层，激活函数，损失函数，优化器，模型容器
import torch.optim as optim  # 实现各种优化算法的模块（SGD,Adam）
import torchvision  # 与计算机视觉相关的库，提供一系列用于处理图像和视频数据的工具、数据集、变换和预训练模型
import torchvision.transforms as transforms
from torchvision import models

# 设置随机种子以保持实验结果的一致性
torch.manual_seed(17)  # 控制权重初始化和数据增强中的随机性

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# 定义ResNet-18模型
net = models.resnet18()

# 修改输出层以适应CIFAR-10数据集的10个类别
num_ftrs = net.fc.in_features  # 获取神经网络模型中最后一个全连接层的输入特征数
net.fc = nn.Linear(num_ftrs, 10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 创建了一个交叉熵损失函数的实例
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)  # net.parameters()是返回模型中所有可学习参数的迭代器

# 训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # torch.cuda.is_available()：检查当前系统是否支持CUDA
net.to(device)  # 将模型和数据移到选择的设备上，存储在GPU上且通过GPU计算
num_epochs = 10  # 调整训练周期
for epoch in range(num_epochs):
    net.train()  # 将模型设置为训练模式，启用梯度计算和参数更新，跟踪输入数据的梯度
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):  # enumerate()函数将可迭代对象组合为一个枚举对象，同时返回索引和元素值，0是enumerate()函数的起始索引值，表示索引从0开始
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将模型和数据移到选择的设备上，存储在GPU上且通过GPU计算

        optimizer.zero_grad()  # 将模型参数的梯度清零，每个训练批次之前清零模型参数的梯度
        # PyTorch默认会在每个参数的梯度张量上累积梯度值，而不是替换它们，这是为了支持一些特殊情况，例如梯度裁剪
        outputs = net(inputs)  # 前向传播，计算前向传播时创建计算图
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算损失函数关于模型参数的梯度
        optimizer.step()  # 执行模型参数的更新优化步骤

        running_loss += loss.item()  # 累计每次迭代损失

    print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(trainloader)}")

# 在测试集上评估模型
net.eval()  # 将模型设置为评估模式，仅用于进行前向传播，生成预测结果，而不会对模型进行训练。
correct = 0
total = 0
with torch.no_grad():  # 其内部的代码块中禁用PyTorch自动求导功能
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # 寻找每行的最大值（概率）即其所在位置（类别）
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 计算正确数量
        # item()用于包含单个元素的张量，用于将张量中的单个元素转换为Python标量，方便算数运算

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
