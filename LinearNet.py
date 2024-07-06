import torch
import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, net_input, hidden0, hidden1, net_output):
        super(LinearNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(net_input, hidden0),
            nn.ReLU(),
            nn.Linear(hidden0, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, net_output),
        )

    # 前向传播
    def forward(self, x):
        # 将输入特征扁平化，默认压缩为1维
        x = self.flatten(x)
        # 将扁平化后的特征输入到全连接层，并获取神经网络的输出
        logits = self.linear_relu_stack(x)
        return logits
