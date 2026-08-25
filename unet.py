"""
Implementation of the U-Net following the paper U-Net: Convolutional Networks for Biomedical Image Segmentation and the following repository: 
https://github.com/milesial/Pytorch-UNet/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 

class UNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512,512)
        
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128,64)

        self.outc = nn.Conv2d(64, n_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)
        return out

class DoubleConv(nn.Module): 
    """
    Performs a Convolution followed by SiLU. Twice. 
    (Conv -> SiLU) * 2"""
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), 
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
    
    def forward(self, x): 
        #Recall that x is a tensor of dimensions [num batch, num chan, height, width]
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscales input with a maxpool, then performs a DoubleConv."""
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module): 
    """
    Upscales input with bilinear upsamling, then performs a DoubleConv.
    
    Arguments: 
        in_channels -- input channels to the DoubleConv
        out_channels -- output channels of the DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_channels, out_channels)
        self.in_channels = in_channels
    
    def forward(self, x1, x2): 
        #I AM ONLY CONSIDERING, FOR NOW, THE CASE WHERE INPUT IMAGES ARE OF DIMENSION 2^n x 2^m
        #check the reference repository for details about padding
        
        #we are concatenating around dimension 1, i.e. we are augmenting the number of channels
        assert x1.size(1) + x2.size(1) == self.in_channels, \
        "the number of concatenated channels does not correspond to the number of channels in input to the Conv2d"
        x1 = self.up(x1)
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)