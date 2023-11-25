import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def exists(val):
    return val is not None


# Q learning on robotic transformer

class QLearner(Module):
    raise NotImplementedError
