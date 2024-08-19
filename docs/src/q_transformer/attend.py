from functools import wraps
from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

# helpers

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def maybe_reduce_mask_and(*maybe_masks):
    maybe_masks = [*filter(exists, maybe_masks)]

    if len(maybe_masks) == 0:
        return None

    mask, *rest_masks = maybe_masks

    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        causal = False,
        flash_config: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        if flash:
            print_once('using memory efficient attention')

        self.flash_config = flash_config

    def flash_attn(self, q, k, v, mask = None, attn_mask = None):
        _, heads, q_len, dim_head, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = mask.expand(-1, heads, q_len, -1)

        mask = maybe_reduce_mask_and(mask, attn_mask)

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with torch.backends.cuda.sdp_kernel(**self.flash_config):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                is_causal = self.causal,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v, mask = None, attn_mask = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask, attn_mask = attn_mask)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # causal mask

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = sim.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # key padding mask

        if exists(mask):
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # attention mask

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
