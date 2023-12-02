from pathlib import Path

import numpy as np

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

from einops import rearrange, repeat, reduce

from q_transformer.q_robotic_transformer import QRoboticTransformer

from torchtyping import TensorType

from beartype import beartype
from beartype.typing import Iterator, Tuple

from tqdm import tqdm

# helpers

def exists(v):
    return v is not None

# base environment class to extend

class BaseEnvironment(Module):
    def __init__(
        self,
        state_shape: Tuple[int, ...] = ()
    ):
        super().__init__()
        self.state_shape = state_shape
        self.register_buffer('dummy', torch.zeros(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def init(self) -> Tuple[str, Tensor]: # (instruction, initial state)
        raise NotImplementedError

    def forward(
        self,
        actions: Tensor
    ) -> Tuple[
        TensorType[(), float],     # reward
        Tensor,                    # next state
        TensorType[(), bool]       # done
    ]:
        raise NotImplementedError

# agent class

class Agent(Module):
    @beartype
    def __init__(
        self,
        q_transformer: QRoboticTransformer,
        *,
        environment: BaseEnvironment,
        dataset_folder: str = './dataset',
        num_episodes: int = 1000,
        max_num_steps_per_episode: int = 10000,
        epsilon_start: float = 0.25,
        epsilon_end: float = 0.001,
        num_steps_to_target_epsilon: int = 1000
    ):
        super().__init__()
        self.q_transformer = q_transformer
        self.environment = environment

        assert 0. <= epsilon_start <= 1.
        assert 0. <= epsilon_end <= 1.

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        self.num_episodes = num_episodes
        self.max_num_steps_per_episode = max_num_steps_per_episode
        self.num_steps_to_target_epsilon = num_steps_to_target_epsilon

        self.dataset_folder = Path(dataset_folder)
        self.dataset_folder.mkdir(exist_ok = True, parents = True)
        assert self.dataset_folder.is_dir()

    def get_epsilon(self, step):
        if step >= self.num_steps_to_target_epsilon:
            return self.epsilon_end

        return ((self.epsilon_end - self.epsilon_start) / float(self.num_steps_to_target_epsilon)) * float(step) + self.epsilon_start

    @beartype
    @torch.no_grad()
    def forward(self):
        self.q_transformer.eval()

        for episode in range(self.num_episodes):
            print(f'episode {episode}')

            instruction, curr_state = self.environment.init()

            memories = []

            for step in tqdm(range(self.max_num_steps_per_episode)):
                epsilon = self.get_epsilon(step)

                actions = self.q_transformer.get_actions(
                    rearrange(curr_state, '... -> 1 ...'),
                    [instruction],
                    prob_random_action = epsilon
                )

                reward, next_state, done = self.environment(actions)

                memories.append((curr_state, actions, reward, next_state, done))

        return memories
