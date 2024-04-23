import torch
import torch.nn as nn
from dense_layer import dense
import torch.nn.functional as F
import math
import numpy as np

group_num = 16

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_timestep_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb
    


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

    
# class Discriminator(nn.Module):
#     def __init__(self, in_channels, features=[64, 128, 256]):
#         super().__init__()
#         blocks = []
#         for idx, feature in enumerate(features):
#             blocks.append(BasicBlock(in_channels, feature, (1,1,1), 1, residual=True))
#             blocks.append(nn.MaxPool3d((2, 2, 2)))
#             in_channels = feature
            

#         self.blocks = nn.Sequential(*blocks)
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool3d((1, 1, 1)), # 3D pooling
#             nn.Flatten(),
#             nn.Linear(256, 128), # adjusted linear layer
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(128, 1),
#         )

#     def forward(self, x):
#         x = self.blocks(x)
#         x = self.classifier(x)
#         return x
class BasicBlockDisc(nn.Module):
    def __init__(self, input_dim, output_dim, t_emb_dim, stride=1, padding=1, groups=1):
        super(BasicBlockDisc, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.GroupNorm(groups, output_dim),
            nn.ELU(inplace=True),
            nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=padding),
            nn.GroupNorm(groups, output_dim)
        )
        self.dense_t1= dense(t_emb_dim, output_dim)

       
        self.convskip = nn.Sequential(
            nn.Conv3d(input_dim, output_dim, kernel_size=1, stride=stride),
            nn.GroupNorm(groups, output_dim)
        )


        self.elu = nn.ELU(inplace=True)

    def forward(self, x, t_emb):
        out = self.convblock(x)
        temp = self.dense_t1(t_emb)[..., None, None, None]
        out += temp

        residual = self.convskip(x)

        out += residual
        out = self.elu(out)

        return out

    
class Discriminator(nn.Module):
    def __init__(self, in_channels,features=[64, 128, 256, 512], t_emb_dim = 128, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.features = features
        self.in_channels = in_channels
        self.act = act
        self.t_emb_dim = t_emb_dim
        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
            )
        self.blocks = []
        
        self.conv1 = BasicBlockDisc(self.in_channels, self.features[0], self.t_emb_dim, (1,1,1), 1)
        self.conv2 = BasicBlockDisc(self.features[0], self.features[1], self.t_emb_dim, (1,1,1), 1)
        self.conv3 = BasicBlockDisc(self.features[1], self.features[2], self.t_emb_dim, (1,1,1), 1)
        self.conv4 = BasicBlockDisc(self.features[2], self.features[3], self.t_emb_dim, (1,1,1), 1)
       
        self.maxpool = nn.MaxPool3d((2, 2, 2))

        # Classifier remains the same
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x, t):
        t_embed = self.act(self.t_embed(t)) 

        h0 = self.conv1(x, t_embed)
        h0 = self.maxpool(h0)
 
        h1 = self.conv2(h0, t_embed)
        h1 = self.maxpool(h1)

        h2 = self.conv3(h1, t_embed)
        h2 = self.maxpool(h2)

        h3 = self.conv4(h2, t_embed)
        h3 = self.maxpool(h3)

        out = self.classifier(h3)
        return out

