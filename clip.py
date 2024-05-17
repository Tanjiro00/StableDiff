import numpy as np
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
import torch


class CLIP_Embeddings(nn.Module):
    def __init__(self, len_vocab: int, d_emb: int, n_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(len_vocab, d_emb)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, d_emb))

    
    def forward(self, x):
        embs = self.token_embedding(x)
        embs += self.position_embedding
        return embs


class CLIPLayer(nn.Module):
    def __init__(self, n_heads, d_emb):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(d_emb)
        self.attention = SelfAttention(d_emb, n_heads)
        self.layernorm_2 = nn.LayerNorm(d_emb)
        self.linear_1 = nn.Linear(d_emb, d_emb * 4)
        self.linear_2 = nn.Linear(d_emb * 4, d_emb)
        
    def forward(self, x):
        # (bs, seq_len, emb_dim)
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, mask=True) # mb I need add mask, but it isnt logic
        x += residue
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x) # (bs, seq_len, d_emb) -> (bs, seq_len, d_emb * 4) 
        x = x * torch.sigmoid(1.702 * x) # QuickGeLU https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
        x = self.linear_2(x)# (bs, seq_len, emb_dim)
        x += residue
        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIP_Embeddings(49408, 768, 77)
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])
        self.layernorm = nn.LayerNorm(768)

    
    def forward(self, x) -> torch.FloatTensor:
        x = x.type(torch.long)
        embs = self.embedding(x)# (bs, seq_len) -> (bs, seq_len, emb_dim)
        for clip_layer in self.layers:
            embs = clip_layer(embs)
        out = self.layernorm(embs)
        return out

