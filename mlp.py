import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 超参数
batch_size = 200
learning_rate = 0.01
epochs = 10

# 获取训练数据
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,  # train=True则得到的是训练集
                   transform=transforms.Compose([  # transform进行数据预处理
                       transforms.ToTensor(),  # 转成Tensor类型的数据
                       transforms.Normalize((0.1307,), (0.3081,))  # 进行数据标准化(减去均值除以方差)
                   ])),
    batch_size=batch_size, shuffle=True)  # 按batch_size分出一个batch维度在最前面,shuffle=True打乱顺序

# 获取测试数据
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(  # 定义网络的每一层，nn.ReLU可以换成其他激活函数，比如nn.LeakyReLU()
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class MLP2(nn.Module):

    def __init__(self):
        super(MLP2, self).__init__()
        self.linear0 = nn.Linear(784, 200)
        self.linear1 = nn.Linear(200, 200)
        self.linear2 = nn.Linear(200, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear0(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# yx_attention_pls MLP2 remove ReLU to MLP, the acc is higher than MLP.
net = MLP2()
# 定义sgd优化器,指明优化参数、学习率，net.parameters()得到这个类所定义的网络的参数[[w1,b1,w2,b2,...]
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)  # 将二维的图片数据摊平[样本数,784]

        logits = net(data)  # 前向传播
        loss = criteon(logits, target)  # nn.CrossEntropyLoss()自带Softmax

        optimizer.zero_grad()  # 梯度信息清空
        loss.backward()  # 反向传播获取梯度
        optimizer.step()  # 优化器更新

        if batch_idx % 100 == 0:  # 每100个batch输出一次信息
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0  # correct记录正确分类的样本数
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = net(data)
        test_loss += criteon(logits, target).item()  # 其实就是criteon(logits, target)的值，标量

        pred = logits.data.max(dim=1)[1]  # 也可以写成pred=logits.argmax(dim=1)
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
