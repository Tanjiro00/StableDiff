import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from attention import SelfAttention, CrossAttention


class TimeEmbeddings(nn.Module):
    def __init__(self, n_embedd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embedd, n_embedd * 4)
        self.linear_2 = nn.Linear(n_embedd * 4, n_embedd * 4)
        
    def forward(self, x):
        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        x = F.silu(x)
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)
        # (1, 1280)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, n_embedd_in, n_embedd_out, time_emb=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, n_embedd_in)
        self.conv_feature = nn.Conv2d(n_embedd_in, n_embedd_out, kernel_size=3, padding=1)
        
        self.linear_time = nn.Linear(time_emb, n_embedd_out)
        self.groupnorm_merged = nn.GroupNorm(32, n_embedd_out)
        self.conv_merged = nn.Conv2d(n_embedd_out, n_embedd_out, kernel_size=3, padding=1)
        if n_embedd_in == n_embedd_out:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(n_embedd_in, n_embedd_out, kernel_size=1, padding=0)

    
    def forward(self, x, time):
        # x -> (bs, n_embedd_in, h, w)
        # time -> (1, 1280)
        residual = x
        # print(x.shape)
        x = self.groupnorm_feature(x)
        x = F.silu(x)
        x = self.conv_feature(x)
        time = F.silu(time)
        time_emb = self.linear_time(time)
        # print(time_emb.shape, x.shape)
        merged = time_emb.unsqueeze(-1).unsqueeze(-1) + x 
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        return merged + self.residual_layer(residual)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embedd, emb_context_d=768):
        super().__init__()
        channels = n_embedd * n_head
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(channels, n_head, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(channels, n_head, emb_context_d, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(channels * 4, channels)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    
    def forward(self, x, context):
        # x -> (bs, channels, h, w)
        # context -> (bs, seq_len, emb_dim)
        residue_long = x
        x = self.groupnorm(x)
        x = self.conv_input(x)
        bs_l, ch_l, h_l, w_l = x.shape
        x = x.view(bs_l, ch_l, h_l * w_l)
        x = x.transpose(-1,-2)
        # x -> (bs, h*w, channels)
        
        #LayerNorm + SelfAttention + SkipConnection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        #LayerNorm + CrossAttention + SkipConnection
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short

        #LayerNorm + FFN + GeGLU + SkipConnection
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        x = x.transpose(2, 1)
        x = x.view(bs_l, ch_l, h_l, w_l)
        
        return self.conv_output(x) + residue_long
        


class Upsample(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # (bs, channels, h, w) -> (bs, channels, h * 2, w * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (b, 4, h/8, w/8) -> (bs, 320, h/8, w/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (b, 320, h/8, w/8) -> (bs, 640, h/16, w/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, padding=1, stride=2)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (b, 640, h/8, w/8) -> (bs, 1280, h/16, w/16)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, padding=1, stride=2)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (b, 1280, h/32, w/32) -> (bs, 1280, h/64, w/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, padding=1, stride=2)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40))
        ])

    
    def forward(self, x, context, time):
        skip_connections = []
        for layer in self.encoders:
            x = layer(x, context, time)
            skip_connections.append(x)
        x = self.bottleneck(x, context, time)
        for layer in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, context, time)
        return x 


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

    
    def forward(self, x) -> torch.Tensor:
        # x -> bs 320 h/8 w/8
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        # x -> bs 4 h/8 w/8
        return x




class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbeddings(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        #latent -> (b, 4, h/8, w/8)
        #context -> bs seq_len dim
        #time -> 1, 320
        time = self.time_embedding(time)
        unet_out = self.unet(latent, context, time)
        output = self.final(unet_out)
        return output