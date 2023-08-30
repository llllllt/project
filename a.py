import torch
import torch.nn as nn
# from torchviz import make_dot

# 定义神经网络
class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.input_layer = nn.Linear(2, 1)  # 输入层，2个输入特征，1个输出
        self.activation = nn.ReLU()  # 激活函数
        self.output_layer = nn.Linear(1, 1)  # 输出层，1个输入（来自隐藏层），1个输出

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x

# 创建网络实例
net = SimpleNetwork()

criterion = nn.MSELoss()

a = torch.tensor([[1., 2.],
                  [3., 4.]], requires_grad=True)

b = torch.tensor([[2., 3.],
                  [4., 6.]], requires_grad=True)    

print(criterion(a, b))
