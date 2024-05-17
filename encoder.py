import numpy as np
from torch import nn
from torch.nn import functional as F
from decoder import VAE_ResidualBlock, VAE_AttentionBlock
import torch

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            
            nn.Conv2d(3, 128, kernel_size= 3, padding=1),
            
            VAE_ResidualBlock(128, 128),
            
            VAE_ResidualBlock(128, 128),
            #(bs, 128, h, w) -> (bs, 128, h/2, w/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(128, 256),
            
            VAE_ResidualBlock(256, 256),
            #(bs, 256, h/2, w/2) -> (bs, 256, h/4, w/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(256, 512),
            
            VAE_ResidualBlock(512, 512),
            #(bs, 512, h/4, w/4) -> (bs, 512, h/8, w/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),

            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            nn.Conv2d(8, 8, kernel_size=1, padding=0)
            
        )


    def forward(self, x, noise):
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # делаю паддинг когда у нас есть аргумент stride=2, т.к. мы в этот момент мы хотим чтобы было h/2 w/2, а не h/2-1 w/2 - 1
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        #(b, 8, h/8, w/8) -> (b, 4, h/8, w/8) (b, 4, h/8, w/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, min=-30, max=20)
        variance = log_variance.exp()
        sigma = variance ** 0.5 # may error

        #from N(0, 1) -> N(mean, simga)
        x = mean + noise * sigma

        x *= 0.18215

        return x
                