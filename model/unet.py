import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        # conv_down 1
        self.conv_down_1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv_down_1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # conv_down 2
        self.conv_down_2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_down_2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # conv_down 3
        self.conv_down_3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_down_3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # conv_down 4
        self.conv_down_4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv_down_4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  
   
        # conv_up 3
        self.conv_up_3_1 = nn.Conv2d(256 + 512, 256, kernel_size=3, padding=1)
        self.conv_up_3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  

        # conv_up 2
        self.conv_up_2_1 = nn.Conv2d(128 + 256, 128, kernel_size=3, padding=1)
        self.conv_up_2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 

        # conv_up 1
        self.conv_up_1_1 = nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1)
        self.conv_up_1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 

        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)     
        
        
    def forward(self, x):

        # conv_down 1
        conv_down_1_1 = self.conv_down_1_1(x)
        x = F.relu(conv_down_1_1, inplace=True)
        conv_down_1_2 = self.conv_down_1_2(x)
        conv1 = F.relu(conv_down_1_2, inplace=True)
        x = self.maxpool(conv1)

        # conv_down 2
        conv_down_2_1 = self.conv_down_2_1(x)
        x = F.relu(conv_down_2_1, inplace=True)
        conv_down_2_2 = self.conv_down_2_2(x)
        conv2 = F.relu(conv_down_2_2, inplace=True)
        x = self.maxpool(conv2)

        # conv_down 3
        conv_down_3_1 = self.conv_down_3_1(x)
        x = F.relu(conv_down_3_1, inplace=True)
        conv_down_3_2 = self.conv_down_3_2(x)
        conv3 = F.relu(conv_down_3_2, inplace=True)
        x = self.maxpool(conv3)

        # conv_down 4
        conv_down_4_1 = self.conv_down_4_1(x)
        x = F.relu(conv_down_4_1, inplace=True)
        conv_down_4_2 = self.conv_down_4_2(x)
        x = F.relu(conv_down_4_2, inplace=True)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        # conv_up 3
        conv_up_3_1 = self.conv_up_3_1(x)
        x = F.relu(conv_up_3_1, inplace=True)
        conv_up_3_2 = self.conv_up_3_2(x)
        x = F.relu(conv_up_3_2, inplace=True)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)

        # conv_up 2
        conv_up_2_1 = self.conv_up_2_1(x)
        x = F.relu(conv_up_2_1, inplace=True)
        conv_up_2_2 = self.conv_up_2_2(x)
        x = F.relu(conv_up_2_2, inplace=True)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        # conv_up 1
        conv_up_1_1 = self.conv_up_1_1(x)
        x = F.relu(conv_up_1_1, inplace=True)
        conv_up_1_2 = self.conv_up_1_2(x)
        x = F.relu(conv_up_1_2, inplace=True)
        
        # last
        out = self.conv_last(x)
        
        return out
    