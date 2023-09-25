# 导入所需库文件
import torch.nn as nn


# 定义ResNet中的基本块（残差模块）
class BasicBlock(nn.Module):  # 定义一个继承自nn.Module的类，使BasicBlock成为一个PyTorch模块，是定义自定义神经网络模块的标准方式，使模块能够与PyTorch生态系统集成
    # 定义类属性(数据属性和函数属性)
    # 数据属性：变量，存储值
    expansion = 1  # 类的一个属性，表示每个基本块的通道扩展系数：输出通道数/输入通道数

    # 函数属性：类函数(方法)，执行操作
    def __init__(self, in_channels, out_channels, stride=1):  # 接受三个参数：输入特征图通道数；输出特征图通道数；卷积步幅（默认为1）
        # 类首先都要有一个构造函数
        # 函数__init__是BasicBlock类的构造函数，用于在创建类的实例时，初始化该实例的属性和状态。
        # self总是第一个参数，作为指向当前实例的引用，允许在类的方法内部访问和操作实例的属性和方法（方便初始化）
        super(BasicBlock, self).__init__()
        # super()函数返回一个代表父类的对象，__init__()是其构造函数
        # 在子类‘BasicBlock’中调用父类‘nn.Module’的构造函数，以确保正确初始化BasicBlock实例，并且能够利用nn.Module类的功能，即子类能够正确地继承和扩展父类的功能
        # 调用父类的构造函数来执行以下任务：确保父类的属性和方法被正确初始化，让父类执行一些必要的初始化工作，允许子类扩展或修改父类的行为
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # 卷积层
        # 使用self.方法可以将函数作为BasicBlock类的一个属性来存储，提供更好的代码组织和封装、属性访问和继承能力
        # 属性封装：将函数作为属性存储在类中，使得它可以在类的其他方法中访问。同时提高代码的可读性，可以明确地看到conv1是BasicBlock类的一部分
        # 属性命名空间：将函数的名称限定在类的命名空间内，避免与其他可能在同一代码中定义的变量或对象冲突
        # 属性继承：如果BasicBlock类是其他类的基类，那么self.conv1将在子类中继续存在，使得子类可以重用和扩展BasicBlock类的功能。
        # 属性访问权限：明确地表示conv1是BasicBlock类的一个属性，而不是全局变量
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批量归一化层，参数表示要归一化的特征通道数，通常与卷积层的输出通道数相同
        # 对多通道输出分别做批量归一化，每个通道都有独立的拉伸和偏移参数。对单通道：batch为m、卷积输出(p,q)，对该通道中m×p×q个元素同时做批量归一化,使用相同的均值和方差
        # BN层通常被添加到卷积层或全连接层之后，规范网络中间层的输出、提高层间独立性
        # self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)  # 激活函数，参数表示是否在原地修改输入，设置为True会直接修改输入数据而不创建新的张量，有助于节省内存
        # 激活函数通常被添加到卷积层或全连接层的后面，以增加网络的非线性性
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # 卷积层
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批量归一化层

        self.downsample = None  # 下采样（可选）
        if stride != 1 or in_channels != out_channels:  # 处理输入输出通道数不一致情况。残差连接支路，输入输出一致才能相加
            if stride == 2:
                self.downsample = nn.Sequential(  # 层顺序定义法定义网络层
                    nn.AvgPool2d(kernel_size=2, stride=2),  # 防止信息丢失
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),  # 卷积层 1×1卷积可用于更改通道数
                    nn.BatchNorm2d(out_channels)  # 批归一化
                )
            else:
                self.downsample = nn.Sequential(  # 层顺序定义法定义网络层
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  # 卷积层 1×1卷积可用于更改通道数
                    nn.BatchNorm2d(out_channels)  # 批归一化
                )

    def forward(self, x):  # 定义前向传播函数，定义了当数据通过BasicBlock类对象时，数据应该如何在模块内前向传播
        # forward()函数的默认行为是在基类nn.Module中定义的，但默认行为并不执行任何有用的计算，需要在自定义类中重写
        residual = x  # 残差连接支路

        # 按顺序依次连接各层
        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.dropout(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:  # 下采样判断
            residual = self.downsample(x)

        out += residual  # 残差连接，将输出与原始输入相加，是ResNet中的关键部分，有助于防止梯度消失问题，使网络更容易训练
        out = self.relu(out)

        return out


# 定义ResNet-18模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):  # 构造函数
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大值池化层
        self.layer1 = self.make_layer(BasicBlock, 64, 2, stride=1)  # 残差模块
        self.layer2 = self.make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层，参数(1,1)为池化后输出尺寸
        # 自适应：输出尺寸不是通过预定义的池化窗口大小和步幅来指定，而是通过指定输出的目标尺寸来自适应地计算
        self.fc = nn.Linear(512, num_classes)  # 全连接层，进行最后的分类输出

    def make_layer(self, block, out_channels, num_blocks, stride):  # 创建一个由多个相同类型的基本块组成的层
        # num_blocks：要创建的基本块的数量，即这一层包含多少个基本块
        strides = [stride] + [1] * (num_blocks - 1)  # 创建一个列表 [stride, 1, 1,···, 1]，除第一个外剩下的通常都是1
        layers = []
        for stride in strides:  # 依次读取各层的stride值，依次添加基本块到层中
            layers.append(block(self.in_channels, out_channels, stride))  # 添加一个基本块，构造函数创建BasicBlock实例
            self.in_channels = out_channels * block.expansion  # 更新下一个基本块的输入通道数
        return nn.Sequential(*layers)  # 层顺序定义法定义网络层
        # ‘*’操作符，可将列表中的元素解包成一个个单独的参数，传递给nn.Sequential的构造函数作为单独的参数，而不是一个包含多个基本块的列表

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv_1(x)
        # x = self.conv_2(x)
        # x = self.conv_3(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 对张量通过view函数进行reshape，输入参数为元组，x.size(0)通常表示批处理维度，-1表示自动计算维度大小
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":  # 只有作为main函数时才会被调用，防止import时被执行
    # 创建ResNet-18模型实例
    resnet18 = ResNet18()  # 调用ResNet18的构造函数，创建ResNet18的一个实例
    # 使用类的名称和括号来创建实例，ResNet18()即可，不需要显式调用构造函数的名称，不需要显式传递nn.Module父类，需要时可以传入初始参数
    # resnet18 = torchvision.models.resnet18()，另一种方法：直接使用pytorch中定义好的模型来构建网络

    # output = resnet18(input)，当你调用一个模型实例并传递输入数据作为参数时，实际上是在执行该模型的前向传播操作，执行了forward()函数
    # resnet18(input)是执行模型前向传播的操作，而resnet18()是创建模型实例的操作

    # 打印模型结构
    print(resnet18)
