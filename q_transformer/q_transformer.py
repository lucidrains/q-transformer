import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

from einops import rearrange
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Union, List, Tuple

from q_transformer.robotic_transformer import QRoboticTransformer

# helpers

def exists(val):
    return val is not None

# Q learning on robotic transformer

class QLearner(Module):

    @beartype
    def __init__(
        self,
        model: Union[QRoboticTransformer, Module],
        *,
        dataset: Dataset
    ):
        super().__init__()
        assert model.num_actions == 1

        self.model = model

    def forward(self, x):
        return x
