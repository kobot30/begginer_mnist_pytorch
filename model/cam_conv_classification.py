import torch.nn as nn


class CamConvClassification(nn.Module):
    def __init__(self):
        super(CamConvClassification, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(1, out_channels=16, kernel_size=3, padding=1), # 28x28 pixcel
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 14x14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 14 * 14, 1000), # out_channels * pixcel * pixcel
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size()[0], -1)
        out = self.fc_layer(out)
        return out
