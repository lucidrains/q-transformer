from random import random
from functools import partial, cache

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from beartype import beartype
from beartype.typing import Union, List, Optional, Callable, Tuple, Dict, Any

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

from q_transformer.attend import Attend

from classifier_free_guidance_pytorch import (
    TextConditioner,
    AttentionTextConditioner,
    NullConditioner,
    classifier_free_guidance
)

# helpers

def exists(val):
    return val is not None

def xnor(x, y):
    """ (True, True) or (False, False) -> True """
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# tensor helpers

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim)

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

# 2d rotary positional embedding
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(Module):
    def __init__(self, dim, omega = 10000):
        super().__init__()
        inv_freq = 1.0 / (omega ** (torch.arange(0, dim, 4).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    @autocast(enabled = False)
    def forward(self, height_width):
        device, dtype = self.inv_freq.device, self.inv_freq.dtype

        axial_pos = torch.arange(height_width, device = device).type(dtype)

        freqs = torch.einsum('i, j -> i j', axial_pos, self.inv_freq)
        freqs = repeat(freqs, '... f -> ... (f c)', c = 2)

        freqs = torch.broadcast_tensors(freqs[None, :, :], freqs[:, None, :])
        freqs = torch.cat(freqs, dim = -1)
        return rearrange(freqs, '... f -> (...) f')

def rotate_half(x):
    x1, x2 = rearrange(x, '... (d c) -> ... d c', c = 2).unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d c -> ... (d c)')

@autocast(enabled = False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# sync batchnorm

@cache
def get_is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def MaybeSyncBatchnorm2d(is_distributed = None):
    is_distributed = default(is_distributed, get_is_distributed())
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm2d

# channel rmsnorm

class RMSNorm(Module):
    def __init__(self, dim, affine = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x):
        return l2norm(x) * self.gamma * self.scale

class ChanRMSNorm(Module):
    def __init__(self, dim, affine = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1)) if affine else 1.

    def forward(self, x):
        return l2norm(x, dim = 1) * self.gamma * self.scale

# sinusoidal positions

def posemb_sincos_1d(seq, dim, temperature = 10000, device = None, dtype = torch.float32):
    n = torch.arange(seq, device = device)
    omega = torch.arange(dim // 2, device = device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim = 1)
    return pos_emb.type(dtype)

# helper classes

class Residual(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.,
        adaptive_ln = False
    ):
        super().__init__()
        self.adaptive_ln = adaptive_ln

        inner_dim = int(dim * mult)
        self.norm = RMSNorm(dim, affine = not adaptive_ln)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        cond_fn: Optional[Callable] = None
    ):
        x = self.norm(x)

        assert xnor(self.adaptive_ln, exists(cond_fn))

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)

# MBConv

class SqueezeExcitation(Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)

class MBConvResidual(Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        batch, device = x.shape[0], x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((batch, 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.,
    is_distributed = None,
    use_layernorm = True
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    if use_layernorm:
        norm_klass = ChanRMSNorm
    else:
        norm_klass = MaybeSyncBatchnorm2d(is_distributed)

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        norm_klass(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        norm_klass(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        norm_klass(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 32,
        dropout = 0.,
        window_size = 7,
        num_mem_kv = 4,
        flash = True
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.heads = heads

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, self.heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
        )

        self.attend = Attend(
            causal = False,
            dropout = dropout,
            flash = flash
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rotary_emb = None
    ):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x = self.norm(x)

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        g = self.to_v_gates(x)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 2d rotary

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        # attention

        out = self.attend(q, k, v)

        # gate values per head, allow for attending to nothing

        out = out * g

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(Module):
    @beartype
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth: Tuple[int, ...],
        heads = 8,
        dim_head = 64,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        use_layernorm = True,
        dropout = 0.1,
        channels = 3,
        flash_attn = True
    ):
        super().__init__()
        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1)
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        self.window_size = window_size
        w = window_size

        # rotary embedding

        assert divisible_by(dim_head, 4), f'{dim_head} must be divisible by 4 for axial rotary embedding for maxvit'

        self.axial_rotary_emb = RotaryEmbedding(dim_head)
        self.register_buffer('cached_rotary_emb', self.axial_rotary_emb(window_size), persistent = False)

        # iterate through stages

        cond_hidden_dims = []

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.ModuleList([
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate,
                        use_layernorm = use_layernorm
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
                    Residual(Attention(dim = layer_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = w, flash = flash_attn)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),

                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
                    Residual(Attention(dim = layer_dim, heads = heads, dim_head = dim_head, dropout = dropout, window_size = w, flash = flash_attn)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                ])

                self.layers.append(block)

        embed_dim = dims[-1]
        self.embed_dim = dims[-1]

        self.cond_hidden_dims = cond_hidden_dims

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    @beartype
    def forward(
        self,
        img,
        texts: Optional[List[str]] = None,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        cond_drop_prob = 0.,
        return_embeddings = False
    ):
        assert all([divisible_by(d, self.window_size) for d in img.shape[-2:]])

        x = self.conv_stem(img)

        rotary_emb = self.cached_rotary_emb

        cond_fns = iter(default(cond_fns, []))

        for (
            mb_conv,
            rearr_windowed_in,
            windowed_attn,
            windowed_ff,
            rearr_windowed_out,
            rearr_grid_in,
            grid_attn,
            grid_ff,
            rearr_grid_out
        ) in self.layers:
            cond_fn = next(cond_fns, None)

            if exists(cond_fn):
                x = cond_fn(x)

            x = mb_conv(x)
            x = rearr_windowed_in(x)
            x = windowed_attn(x, rotary_emb = rotary_emb)
            x = windowed_ff(x)
            x = rearr_windowed_out(x)

            x = rearr_grid_in(x)
            x = grid_attn(x, rotary_emb = rotary_emb)
            x = grid_ff(x)
            x = rearr_grid_out(x)

        if return_embeddings:
            return x

        return self.mlp_head(x)

# attention

class TransformerAttention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        num_mem_kv = 4,
        norm_context = False,
        adaptive_ln = False,
        dropout = 0.1,
        flash = True,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.adaptive_ln = adaptive_ln
        self.norm = RMSNorm(dim, affine = not adaptive_ln)

        self.context_norm = RMSNorm(dim_context) if norm_context else None

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = None
        if num_mem_kv > 0:
            self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))

        self.attend = Attend(
            dropout = dropout,
            flash = flash,
            causal = causal
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None,
        cond_fn: Optional[Callable] = None
    ):
        b = x.shape[0]

        assert xnor(exists(context), exists(self.context_norm))

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        assert xnor(exists(cond_fn), self.adaptive_ln)

        if exists(cond_fn):
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        if exists(self.mem_kv):
            mk, mv = map(lambda t: repeat(t, '... -> b ...', b = b), self.mem_kv)

            k = torch.cat((mk, k), dim = -2)
            v = torch.cat((mv, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (self.num_mem_kv, 0), value = True)

            if exists(attn_mask):
                attn_mask = F.pad(attn_mask, (self.num_mem_kv, 0), value = True)

        out = self.attend(q, k, v, mask = mask, attn_mask = attn_mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.,
        adaptive_ln = False,
        flash_attn = True,
        cross_attend = False,
        causal = False,
        final_norm = True
    ):
        super().__init__()
        self.layers = ModuleList([])

        attn_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            dropout = attn_dropout,
            flash = flash_attn
        )

        for _ in range(depth):
            self.layers.append(ModuleList([
                TransformerAttention(**attn_kwargs, causal = causal, adaptive_ln = adaptive_ln, norm_context = False),
                TransformerAttention(**attn_kwargs, norm_context = True) if cross_attend else None,
                FeedForward(dim = dim, dropout = ff_dropout, adaptive_ln = adaptive_ln)
            ]))

        self.norm = RMSNorm(dim) if final_norm else nn.Identity()

    @beartype
    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None,
        context: Optional[Tensor] = None
    ):
        cond_fns = iter(default(cond_fns, []))

        for attn, maybe_cross_attn, ff in self.layers:
             x = attn(x, attn_mask = attn_mask, cond_fn = next(cond_fns, None)) + x

             if exists(maybe_cross_attn):
                assert exists(context)
                x = maybe_cross_attn(x, context = context) + x

             x = ff(x, cond_fn = next(cond_fns, None)) + x

        return self.norm(x)

# token learner module

class TokenLearner(Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        num_layers = 2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups = num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g c) h w', g = self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g = self.num_output_tokens)

        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '* c n')
        return x

# Dueling heads for Q value

class DuelingHead(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 2,
        action_bins = 256
    ):
        super().__init__()
        dim_hidden = dim * expansion_factor

        self.stem = nn.Sequential(
            nn.Linear(dim, dim_hidden),
            nn.SiLU()
        )

        self.to_values = nn.Sequential(
            nn.Linear(dim_hidden, 1)
        )

        self.to_advantages = nn.Sequential(
            nn.Linear(dim_hidden, action_bins)
        )

    def forward(self, x):
        x = self.stem(x)

        advantages = self.to_advantages(x)
        advantages = advantages - reduce(advantages, '... a -> ... 1', 'mean')

        values = self.to_values(x)

        q_values = values + advantages
        return q_values.sigmoid()

# Q head modules, for either single or multiple actions

class QHeadSingleAction(Module):
    def __init__(
        self,
        dim,
        *,
        num_learned_tokens = 8,
        action_bins = 256,
        dueling = False
    ):
        super().__init__()
        self.action_bins = action_bins

        if dueling:
            self.to_q_values = nn.Sequential(
                Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
                DuelingHead(
                    dim,
                    action_bins = action_bins
                )
            )
        else:
            self.to_q_values = nn.Sequential(
                Reduce('b (f n) d -> b d', 'mean', n = num_learned_tokens),
                RMSNorm(dim),
                nn.Linear(dim, action_bins),
                nn.Sigmoid()
            )

    def get_random_actions(self, batch_size):
        return torch.randint(0, self.action_bins, (batch_size,), device = self.device)

    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values = False,
        actions = None,
        **kwargs
    ):
        assert not exists(actions), 'single actions will never receive previous actions'

        q_values = self.forward(encoded_state)

        max_q, action_indices = q_values.max(dim = -1)

        if not return_q_values:
            return action_indices

        return action_indices, max_q

    def forward(self, encoded_state):
        return self.to_q_values(encoded_state)

class QHeadMultipleActions(Module):
    def __init__(
        self,
        dim,
        *,
        num_actions = 8,
        action_bins = 256,
        attn_depth = 2,
        attn_dim_head = 32,
        attn_heads = 8,
        dueling = False
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_bins = action_bins

        self.action_bin_embeddings = nn.Parameter(torch.zeros(num_actions, action_bins, dim))
        nn.init.normal_(self.action_bin_embeddings, std = 0.02)

        self.transformer = Transformer(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            cross_attend = True,
            adaptive_ln = False,
            causal = True,
            final_norm = True
        )

        self.final_norm = RMSNorm(dim)

        self.dueling = dueling
        if dueling:
            self.to_values = nn.Parameter(torch.zeros(num_actions, dim))

    @property
    def device(self):
        return self.action_bin_embeddings.device

    def maybe_append_actions(self, sos_tokens, actions: Optional[Tensor] = None):
        if not exists(actions):
            return sos_tokens

        batch, num_actions = actions.shape
        action_embeddings = self.action_bin_embeddings[:num_actions]

        action_embeddings = repeat(action_embeddings, 'n a d -> b n a d', b = batch)
        past_action_bins = repeat(actions, 'b n -> b n 1 d', d = action_embeddings.shape[-1])

        bin_embeddings = action_embeddings.gather(-2, past_action_bins)
        bin_embeddings = rearrange(bin_embeddings, 'b n 1 d -> b n d')

        tokens, _ = pack((sos_tokens, bin_embeddings), 'b * d')
        tokens = tokens[:, :self.num_actions] # last action bin not needed for the proposed q-learning
        return tokens

    def get_q_values(self, embed):
        num_actions = embed.shape[-2]
        action_bin_embeddings = self.action_bin_embeddings[:num_actions]

        if self.dueling:
            advantages = einsum('b n d, n a d -> b n a', embed, action_bin_embeddings)

            values = einsum('b n d, n d -> b n', embed, self.to_values[:num_actions])
            values = rearrange(values, 'b n -> b n 1')

            q_values = values + (advantages - reduce(advantages, '... a -> ... 1', 'mean'))
        else:
            q_values = einsum('b n d, n a d -> b n a', embed, action_bin_embeddings)

        return q_values.sigmoid()

    def get_random_actions(self, batch_size):
        return torch.randint(0, self.action_bins, (batch_size, self.num_actions), device = self.device)

    @torch.no_grad()
    def get_optimal_actions(
        self,
        encoded_state,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        **kwargs
    ):
        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')
        tokens = self.maybe_append_actions(sos_token, actions = actions)

        action_bins = []

        for action_idx in range(self.num_actions):
            embed = self.transformer(tokens, context = encoded_state)

            last_embed = embed[:, action_idx]
            bin_embeddings = self.action_bin_embeddings[action_idx]

            q_values = einsum('b d, a d -> b a', last_embed, bin_embeddings)

            selected_action_bins = q_values.argmax(dim = -1)
            next_action_embed = bin_embeddings[selected_action_bins]

            tokens, _ = pack((tokens, next_action_embed), 'b * d')

            action_bins.append(selected_action_bins)

        action_bins = torch.stack(action_bins, dim = -1)

        if not return_q_values:
            return action_bins

        all_q_values = self.get_q_values(embed)
        return action_bins, all_q_values

    def forward(
        self,
        encoded_state: Tensor,
        actions: Optional[Tensor] = None
    ):
        """
        einops
        b - batch
        n - number of actions
        a - action bins
        d - dimension
        """

        # this is the scheme many hierarchical transformer papers do

        sos_token = reduce(encoded_state, 'b ... d -> b 1 d', 'mean')

        tokens = self.maybe_append_actions(sos_token, actions = actions)

        embed = self.transformer(tokens, context = encoded_state)

        return self.get_q_values(embed)

# Robotic Transformer

class QRoboticTransformer(Module):

    @beartype
    def __init__(
        self,
        *,
        vit: Union[Dict[str, Any], MaxViT],
        num_actions = 8,
        action_bins = 256,
        depth = 6,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        cond_drop_prob = 0.2,
        use_attn_conditioner = False,
        conditioner_kwargs: dict = dict(),
        dueling = False,                       # https://arxiv.org/abs/1511.06581
        flash_attn = True,
        condition_on_text = True,
        q_head_attn_kwargs: dict = dict(
            attn_heads = 8,
            attn_dim_head = 64,
            attn_depth = 2
        )
    ):
        super().__init__()

        # vit

        if isinstance(vit, dict):
            vit = MaxViT(**vit)

        self.vit = vit

        self.num_vit_stages = len(vit.cond_hidden_dims)

        attend_dim = vit.embed_dim

        # q-transformer related action embeddings

        assert num_actions >= 1

        self.num_actions = num_actions
        self.is_single_action = num_actions == 1
        self.action_bins = action_bins

        # conditioning

        self.condition_on_text = condition_on_text

        if condition_on_text:
            conditioner_klass = AttentionTextConditioner if use_attn_conditioner else TextConditioner

            self.conditioner = conditioner_klass(
                hidden_dims = (*tuple(vit.cond_hidden_dims), *((attend_dim,) * depth * 2)),
                hiddens_channel_first = (*((True,) * self.num_vit_stages), *((False,) * depth * 2)),
                cond_drop_prob = cond_drop_prob,
                **conditioner_kwargs
            )
        else:
            self.conditioner = NullConditioner(hidden_dims = tuple())

        self.token_learner = TokenLearner(
            dim = vit.embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth

        self.transformer = Transformer(
            dim = attend_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth,
            flash_attn = flash_attn,
            adaptive_ln = condition_on_text,
            final_norm = True
        )

        self.cond_drop_prob = cond_drop_prob

        # Q head

        if self.is_single_action:
            self.q_head = QHeadSingleAction(
                attend_dim,
                num_learned_tokens = self.num_learned_tokens,
                action_bins = action_bins,
                dueling = dueling
            )
        else:
            self.q_head = QHeadMultipleActions(
                attend_dim,
                action_bins = action_bins,
                dueling = dueling,
                **q_head_attn_kwargs
            )

    @property
    def device(self):
        return next(self.parameters()).device

    def get_random_actions(self, batch_size = 1):
        return self.q_head.get_random_actions(batch_size)

    @beartype
    def embed_texts(self, texts: List[str]):
        return self.conditioner.embed_texts(texts)

    @torch.no_grad()
    def get_optimal_actions(
        self,
        *args,
        return_q_values = False,
        actions: Optional[Tensor] = None,
        **kwargs
    ):
        encoded_state = self.encode_state(*args, **kwargs)
        return self.q_head.get_optimal_actions(encoded_state, return_q_values = return_q_values, actions = actions)

    def get_actions(
        self,
        video,
        *args,
        prob_random_action = 0.,  # otherwise known as epsilon in RL
        **kwargs,
    ):
        batch_size = video.shape[0]
        assert 0. <= prob_random_action <= 1.

        if random() < prob_random_action:
            return self.get_random_actions(batch_size = batch_size)

        return self.get_optimal_actions(video, *args, **kwargs)

    def encode_state(
        self,
        video: Tensor,
        texts: Optional[Union[List[str], Tuple[str]]] = None,
        text_embeds: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        cond_drop_prob = 0.,
    ):
        """
        einops
        b - batch
        c - channels
        f - frames
        h - height
        w - width
        n - number of learned tokens
        """

        if not self.condition_on_text:
            assert (not exists(texts) and not exists(text_embeds)), 'neither texts nor text embeds should be passed in'
        else:
            assert exists(texts) ^ exists(text_embeds), 'either texts or text embeds must be passed in if conditioning on instructions'

        if exists(texts) and isinstance(texts, tuple):
            texts = list(texts)

        text_cond_kwargs = dict(texts = texts, text_embeds = text_embeds)

        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        frames, device = video.shape[2], video.device

        cond_fns, _ = self.conditioner(
            **text_cond_kwargs,
            cond_drop_prob = cond_drop_prob,
            repeat_batch = (*((frames,) * self.num_vit_stages), *((1,) * self.transformer_depth * 2))
        )

        vit_cond_fns, transformer_cond_fns = cond_fns[:-(depth * 2)], cond_fns[-(depth * 2):]

        video = rearrange(video, 'b c f h w -> b f c h w')
        images, packed_shape = pack_one(video, '* c h w')

        tokens = self.vit(
            images,
            texts = texts,
            cond_fns = vit_cond_fns,
            cond_drop_prob = cond_drop_prob,
            return_embeddings = True
        )

        tokens = unpack_one(tokens, packed_shape, '* c h w')
        learned_tokens = self.token_learner(tokens)

        tokens_per_frame = learned_tokens.shape[-1]
        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')

        # causal attention mask

        attn_mask = ~torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)

        # sinusoidal positional embedding

        pos_emb = posemb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)

        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)

        # attention

        attended_tokens = self.transformer(learned_tokens, cond_fns = transformer_cond_fns, attn_mask = attn_mask)

        return attended_tokens

    @classifier_free_guidance
    def forward(
        self,
        video: Tensor,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,
        cond_drop_prob = 0.,
    ):

        encoded_state = self.encode_state(
            video = video,
            texts = texts,
            text_embeds = text_embeds,
            actions = actions,
            cond_drop_prob = cond_drop_prob
        )

        # head that returns the q values
        # supporting both single and multiple actions

        if self.is_single_action:
            assert not exists(actions), 'actions should not be passed in for single action robotic transformer'

            q_values = self.q_head(encoded_state)
        else:
            q_values = self.q_head(encoded_state, actions = actions)

        return q_values
