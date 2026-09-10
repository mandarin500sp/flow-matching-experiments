"""
Implementation of the U-Net following the paper U-Net: Convolutional Networks for Biomedical Image Segmentation and the following repository: 
https://github.com/milesial/Pytorch-UNet/
The classical U-Net is modified for image generation purposes. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math 
from utils import get_time_embeddings


class UNet_VF(nn.Module): 
    """
    Starting from a UNet, we integrate time into the model, through a Time Embedding. 
    References are the papers
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        Attention is all you need (for the Time Embedding)
    and the following repositories:
        https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2025/diffusion/demo/flow_match.ipynb
        https://github.com/explainingai-code/DDPM-Pytorch/blob/main/models/unet_base.py
    """
    def __init__(self, n_channels, time_emb_dim): 
        super().__init__()

        self.n_channels = n_channels
        self.time_emb_dim = time_emb_dim

        #Nonlinear embedding for time embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2), 
            nn.SiLU(), 
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        #Initial DoubleConv
        self.inc = DoubleConv(n_channels, 64, time_emb_dim)

        #Down
        self.down1 = Down(64, 128, time_emb_dim)
        self.down2 = Down(128, 256, time_emb_dim)
        self.down3 = Down(256, 512, time_emb_dim)
        self.down4 = Down(512,512, time_emb_dim)
        
        #Up
        self.up1 = Up(1024, 256, time_emb_dim)
        self.up2 = Up(512, 128, time_emb_dim)
        self.up3 = Up(256, 64, time_emb_dim)
        self.up4 = Up(128,64, time_emb_dim)

        #Final DoubleConv
        self.outc = nn.Conv2d(64, n_channels, kernel_size=1)
    
    def forward(self, x, time_steps): 
        assert x.size(0) == time_steps.size(0), "batch size and time steps number should be equal"
        
        t_embs = get_time_embeddings(time_steps, self.time_emb_dim, scale_factor=1000)
        t_embs = self.time_mlp(t_embs)
        
        #Encode with time injection, then decode
        x1 = self.inc(x, t_embs)
        x2 = self.down1(x1, t_embs)
        x3 = self.down2(x2, t_embs)
        x4 = self.down3(x3, t_embs)
        x5 = self.down4(x4, t_embs)

        x_up = self.up1(x5, x4, t_embs)
        x_up = self.up2(x_up, x3, t_embs)
        x_up = self.up3(x_up, x2, t_embs)
        x_up = self.up4(x_up, x1, t_embs)

        #Output
        field = self.outc(x_up)
        return field


class DoubleConv(nn.Module): 
    """
    Performs a Convolution followed by a GN and a SiLU. Then, injects time in the embedding by projecting it onto the embedding's channels and applying a Conv + SiLU + Activation to it.  
    (Conv -> SiLU) * 2
    """
    def __init__(self, in_channels, out_channels, time_emb_dim): 
        super().__init__()
        #Conv -> Norm -> SiLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU() 
        )
        #Time projector onto current subspace
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        #Conv -> Norm -> SiLU
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

    def forward(self, x, t_emb): 
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None] #projecting time onto embedding's channels
        h = self.conv2(h)
        return h


class Down(nn.Module):
    """Downscales input with a maxpool, then performs a DoubleConv."""
    def __init__(self, in_channels, out_channels, time_emb_dim): 
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)
    
    def forward(self, x, t_emb):
        return self.conv(self.maxpool(x), t_emb)


class Up(nn.Module): 
    """
    Upscales input with bilinear upsamling, then performs a DoubleConv.
    
    Arguments: 
        in_channels -- input channels to the DoubleConv
        out_channels -- output channels of the DoubleConv
    
    Notice that in_channels == n iff x2 + x1 == n
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)
        self.in_channels = in_channels
    
    def forward(self, x1, x2, t_emb): 
        #I AM ONLY CONSIDERING, FOR NOW, THE CASE WHERE INPUT IMAGES ARE OF DIMENSION 2^n x 2^m
        #check the reference repository for details about padding
        
        assert x1.size(1) + x2.size(1) == self.in_channels, \
        "the number of concatenated channels does not correspond to the number of channels in input to the Conv2d"
        x1 = self.up(x1)
        #concatenating around dimension 1, i.e. augmenting the number of channel
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x, t_emb)