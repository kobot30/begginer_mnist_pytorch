import torch
from torch import nn
import torch.nn.functional as F


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.l1 = nn.Linear(28 * 28, 200)  # 入力は28x28ピクセル
        self.l2 = nn.Linear(200, 100)
        self.l3 = nn.Linear(100, 50)
        self.l4 = nn.Linear(50, 10)  # class = 10

    def forward(self, x):
        x = torch.reshape(x, (-1, 28 * 28))  # 入力：(N, 1, 28, 28) を (N, 784)にreshape
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = nn.Sigmoid()(self.l4(x))
        return x
