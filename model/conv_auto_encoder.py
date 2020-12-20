from torch import nn
import torch.nn.functional as F


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4,
                               kernel_size=3, padding=1)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16,
                                          kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1,
                                          kernel_size=2, stride=2)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        # Encoder                     # (input) [batch, 1, 28, 28]
        x = F.relu(self.conv1(x))     # [batch, 16, 28, 28]
        x = self.pool(x)              # [batch, 16, 14, 14]
        x = F.relu(self.conv2(x))     # [batch, 4, 14, 14]
        x = self.pool(x)              # [batch ,4, 7, 7]

        # Decoder
        x = F.relu(self.t_conv1(x))   # [batch, 16, 14, 14]
        x = self.t_conv2(x)           # [batch, 1, 28, 28]
        return x
