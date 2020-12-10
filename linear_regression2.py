import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch.functional as F
from torch import nn, optim
from torch.autograd import Variable
import numpy as np

LEARN_RATE = 0.1
# 1.准备数据
x = torch.randn([500, 1])
y_true = x * 0.8 + 3


# 定义线性回归模型

class line_regression(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(line_regression, self).__init__()
        self.linear1 = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        y_out = self.linear1(x)
        return y_out


model = line_regression(1, 1)
if torch.cuda.is_available():
    print('torch.cuda.is_available() ={}'.format(torch.cuda.is_available()))
    model = model.cuda()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)
# # 开启交互模式
plt.ion()
for epoch in range(10):
    print('epoch {}'.format(epoch))
    print('*' * 10)
    running_loss = 0
    for x_input, y_target in zip(list(x), list(y_true)):
        if torch.cuda.is_available():
            x_input = x_input.cuda()
            y_target = y_target.cuda()
            # 1. forward
            y_output = model.forward(x_input)
            # 2. loss
            loss = criterion(y_output, y_target)
            running_loss += loss.item()
            # 3. backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('loss: {:.6f}'.format(running_loss))
    if epoch % 2 == 0:
        plt.scatter(x.numpy(), y_true.numpy(), color="r")
        x1 = x.cuda()
        y_predict = model.forward(x1).data.cpu().numpy()
        plt.plot(x.numpy(), y_predict, color="g")
        #
        plt.pause(0.1)
plt.ioff()
plt.show()
