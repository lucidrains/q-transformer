import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None

# attention

class CausalAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (s h d) -> s b h n d', s = 3, h = heads)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(self, x):
        q, k, v = self.to_qkv(x)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        return self.to_out(out)
