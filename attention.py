from torch import nn
import numpy as np
from torch.nn import functional as F
import math
import torch


class SelfAttention(nn.Module):
    def __init__(self, d_emb: int, n_heads: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_emb, 3 * d_emb, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_emb, d_emb, bias=out_proj_bias)
        self.d_head = d_emb // n_heads
        self.n_heads = n_heads

    
    def forward(self, x, mask=False):
        bs, seq_len, d_emb = x.shape
        att_v_shape = (bs, seq_len, self.n_heads, self.d_head)
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1) # q = (bs, seq_len, d_emb)
        q = q.view(att_v_shape).transpose(1, 2) # q = (bs, n_heads, seq_len, d_head)
        k = k.view(att_v_shape).transpose(1, 2)
        v = v.view(att_v_shape).transpose(1, 2)
        
        
        weights = q @ k.transpose(-1, -2)
        #TODO
        if mask:
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights.masked_fill_(mask, -torch.inf)
        weights = weights / math.sqrt(self.d_head)
        weights =  F.softmax(weights, dim=-1) # weights = (bs, n_heads, seq_len, seq_len)
        att = weights @ v # att = (bs, n_heads, seq_len, d_head)
        att = att.transpose(1, 2)
        output = att.reshape((bs, seq_len, d_emb)) # (bs, seq_len, d_emb) may be use reshape insteed view
        output = self.out_proj(output)
        return output


class CrossAttention(nn.Module):
    def __init__(self, d_emb: int, n_heads: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_emb, d_emb, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_emb, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_emb, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_emb, d_emb, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_emb // n_heads

    
    def forward(self, x_image, y_context):
        #(b, len_seq, emb_dim) <- x and y
        bs, len_seq, emb_dim = x_image.shape
        q = self.q_proj(x_image).view(bs, -1, self.n_heads, self.d_head).transpose(1, 2)
        bs_c, len_seq_c, emb_dim_c = y_context.shape
        k = self.k_proj(y_context).view(bs, -1, self.n_heads, self.d_head).transpose(1, 2)# (bs, n_heads, seq_len, dim_head)
        v = self.v_proj(y_context).view(bs, -1, self.n_heads, self.d_head).transpose(1, 2) 
        weights = q @ k.transpose(-1, -2) # bs n_heads seq_len seq_len
        weights /= math.sqrt(self.d_head)
        weights = F.softmax(weights, dim=-1)
        att = weights @ v # bs n_heads seq_len dim_head
        att = att.transpose(1, 2).contiguous()
        output = self.out_proj(att.view((bs, len_seq, emb_dim)))
        return output
        
        
        
        
        
        
        
        