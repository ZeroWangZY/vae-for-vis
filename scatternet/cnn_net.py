#! -*- coding: utf-8 -*-
"""
cnn结构直接搬马哥的ScatterNet
假设输入图像大小是(1, 112, 112)。如果输入图像大小改了，fc1里的7要改
"""
import torch.nn as nn


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()

        self.conv1 = nn.Sequential(
          nn.Conv2d(
            in_channels = 1,    # 灰度图，单通道
            out_channels = 32,  # 卷积核数
            kernel_size = 7,    # 卷积核大小
            stride = 1,         # 卷积后先保持图像大小不变
            padding = 3,
          ),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 2),
        )

        self.conv2 = nn.Sequential(
          nn.Conv2d(
            in_channels = 32,   
            out_channels = 32,  # 卷积核数
            kernel_size = 3,    # 卷积核大小
            stride = 1,         # 卷积后先保持图像大小不变
            padding = 1,
          ),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 2),
        )

        self.conv3 = nn.Sequential(
          nn.Conv2d(
            in_channels = 32,   
            out_channels = 32,  # 卷积核数
            kernel_size = 3,    # 卷积核大小
            stride = 1,         # 卷积后先保持图像大小不变
            padding = 1,
          ),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 2),
        )

        self.conv4 = nn.Sequential(
          nn.Conv2d(
            in_channels = 32,   
            out_channels = 64,  # 卷积核数
            kernel_size = 3,    # 卷积核大小
            stride = 1,         # 卷积后先保持图像大小不变
            padding = 1,
          ),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size = 2),
        )

        self.fc1 = nn.Sequential(
          nn.Linear(32 * 7 * 7, 64),
          nn.ReLU(),
        )

        self.fc2 = nn.Linear(64, 14)

    def convLayers(self, x):
        ret = self.conv1(x)
        ret = self.conv2(ret)
        # ret = self.conv3(ret)
        # ret = self.conv4(ret)
        return ret

    def fcLayers(self, x):
        ret = self.fc1(x)
        ret = self.fc2(ret)
        return ret

    def forward(self, x):
        x = self.convLayers(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 64 * 7 * 7)
        ret = self.fcLayers(x)
        return ret