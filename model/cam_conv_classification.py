import torch.nn as nn


class CamConvClassification(nn.Module):
    def __init__(self, num_feature=32):
        super(CamConvClassification, self).__init__()
        self.num_feature = num_feature

        self.layer = nn.Sequential(
            nn.Conv2d(1, self.num_feature, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature),
            nn.ReLU(),
            nn.Conv2d(self.num_feature, self.num_feature * 2, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(self.num_feature * 2, self.num_feature * 4, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(self.num_feature * 4, self.num_feature * 8, 3, 1, 1),
            nn.BatchNorm2d(self.num_feature * 8),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.num_feature * 8 * 7 * 7, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(x.size()[0], -1)
        out = self.fc_layer(out)
        return out
