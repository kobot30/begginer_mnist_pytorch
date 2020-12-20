import torch
from torch import nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.e1 = nn.Linear(28 * 28, 200)
        self.e2 = nn.Linear(200, 100)
        self.e3 = nn.Linear(100, 50)

        # Decoder
        self.d1 = nn.Linear(50, 100)
        self.d2 = nn.Linear(100, 200)
        self.d3 = nn.Linear(200, 28 * 28)

    def forward(self, x):
        x = torch.reshape(x, (-1, 28 * 28))

        # Encoder
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = self.e3(x)

        # Decoder
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x)

        x = torch.reshape(x, (-1, 1, 28, 28))
        return x
