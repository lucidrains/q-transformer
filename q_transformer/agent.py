from pathlib import Path

import numpy as np

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

from einops import rearrange, repeat, reduce

from q_transformer.q_robotic_transformer import QRoboticTransformer

from beartype import beartype
from beartype.typing import Iterator

# helpers

def exists(v):
    return v is not None

# agent class

class Agent(Module):
    @beartype
    def __init__(
        self,
        *,
        q_transformer: QRoboticTransformer,
        environment: Iterator,
        dataset_folder: str = './dataset',
        num_episodes: int,
        epsilon_start: float,
        epsilon_end: float,
        num_steps_to_target_epsilon: int
    ):
        super().__init__()
        self.q_transformer = q_transformer
        self.environment = environment

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_steps_to_target_epsilon = num_steps_to_target_epsilon

        self.dataset_folder = Path(dataset_folder)
        self.dataset_folder.mkdir(exist_ok = True, parents = True)
        assert self.dataset_folder.is_dir()

        self.register_buffer('step', torch.tensor(0))

    @beartype
    def forward(self) -> Dataset:
        return None
