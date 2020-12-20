import torch
from torch import nn
import torch.nn.functional as F


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.l1 = nn.Linear(28 * 28, 200)
        self.l2 = nn.Linear(200, 100)
        self.l3 = nn.Linear(100, 50)
        self.l4 = nn.Linear(50, 1)  # output = 1

    def forward(self, x):
        x = torch.reshape(x, (-1, 28 * 28))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
