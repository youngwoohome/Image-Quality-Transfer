import torch
import torch.nn as nn

import torch.nn.functional as F
import math
import numpy as np

group_num = 16
    


class ResidualConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = torch.nn.Sequential(
            torch.nn.GroupNorm(group_num, input_dim),
            torch.nn.ELU(),
            torch.nn.Conv3d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            torch.nn.GroupNorm(group_num, output_dim),
            torch.nn.ELU(),
            torch.nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=1, stride=stride, padding=1),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class BasicBlock(nn.Module):

    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        
        self.convblock = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            torch.nn.GroupNorm(group_num, output_dim),
            torch.nn.ELU(inplace=True),
            torch.nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=padding),
            torch.nn.GroupNorm(group_num, output_dim)
        )

        self.convskip = torch.nn.Sequential(
            torch.nn.Conv3d(input_dim, output_dim, kernel_size=1, stride=stride),
            torch.nn.GroupNorm(group_num, output_dim)
        )
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.convblock(x)

        residual = self.convskip(x)

        out += residual
        out = self.elu(out)

        return out

class Upsample(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel=3, stride=2, padding=1, output_padding=1, bias=True):
        super(Upsample, self).__init__()

        self.upsample = torch.nn.ConvTranspose3d(
            input_dim, output_dim, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        self.elu = torch.nn.ELU()

    def forward(self, x):
        return self.elu(self.upsample(x))


class ResUnet(torch.nn.Module):
    def __init__(self, in_channel=1, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = BasicBlock(in_channel, filters[0], (1,1,1), 1)

        #Downsampling
        self.residual_conv1 = BasicBlock(filters[0], filters[1], (2,2,2), 1)
        self.residual_conv2 = BasicBlock(filters[1], filters[2], (2,2,2), 1)
        self.residual_conv3 = BasicBlock(filters[2], filters[3], (2,2,2), 1)
        self.bridge = BasicBlock(filters[3], filters[3], (1,1,1), 1)

        #Upsampling
        self.upsample0 = Upsample(filters[3], filters[2])
        self.up_residual_conv0 = BasicBlock(filters[2] + filters[2] , filters[2], 1, 1)

        self.upsample1 = Upsample(filters[2], filters[1])
        self.up_residual_conv1 = BasicBlock(filters[1] + filters[1] , filters[1], 1, 1)

        self.upsample2 = Upsample(filters[1], filters[0])
        self.up_residual_conv2 = BasicBlock(filters[0] + filters[0] , filters[0], 1, 1)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv3d(filters[0], filters[0]//2, kernel_size=3, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv3d(filters[0]//2, 1, kernel_size=1),
        )

    def forward(self, x):

        x1 = self.input_layer(x) #32

        x2 = self.residual_conv1(x1) #32->16
  
        x3 = self.residual_conv2(x2) #16->8

        x4 = self.residual_conv3(x3) #8->4
    
        x4_out = self.bridge(x4) #4->4

        x5 = self.upsample0(x4_out) #4 -> 8
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.up_residual_conv0(x5)

        x6 = self.upsample1(x5) #8->16
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.upsample2(x6) #16->32
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.up_residual_conv2(x7)


        out = self.output_layer(x7)

        return out

    
class Discriminator(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(BasicBlock(in_channels, feature, (1,1,1), 1))
            blocks.append(nn.MaxPool3d((2, 2, 2)))
            in_channels = feature
            

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)), # 3D pooling
            nn.Flatten(),
            nn.Linear(512, 128), # adjusted linear layer
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x)
        return x


    
