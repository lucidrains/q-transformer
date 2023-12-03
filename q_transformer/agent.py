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
        memories_dataset_folder: str = './replay_memories_data',
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
        assert epsilon_start >= epsilon_end

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_steps_to_target_epsilon = num_steps_to_target_epsilon
        self.epsilon_slope = (epsilon_end - epsilon_start) / num_steps_to_target_epsilon

        self.num_episodes = num_episodes
        self.max_num_steps_per_episode = max_num_steps_per_episode

        mem_path = Path(memories_dataset_folder)
        self.memories_dataset_folder = mem_path

        mem_path.mkdir(exist_ok = True, parents = True)
        assert mem_path.is_dir()

        states_path = mem_path / 'states.memmap.npy'
        actions_path = mem_path / 'actions.memmap.npy'
        rewards_path = mem_path / 'rewards.memmap.npy'
        dones_path = mem_path / 'dones.memmap.npy'

        prec_shape = (num_episodes, max_num_steps_per_episode)
        num_actions = q_transformer.num_actions
        state_shape = environment.state_shape

        self.states = np.memmap(str(states_path), dtype = 'float32', mode = 'w+', shape = (*prec_shape, *state_shape))
        self.actions = np.memmap(str(actions_path), dtype = 'int', mode = 'w+', shape = (*prec_shape, num_actions))
        self.rewards = np.memmap(str(rewards_path), dtype = 'float32', mode = 'w+', shape = prec_shape)
        self.dones = np.memmap(str(dones_path), dtype = 'bool', mode = 'w+', shape = prec_shape)

    def get_epsilon(self, step):
        return max(self.epsilon_end, self.epsilon_slope * float(step) + self.epsilon_start)

    @beartype
    @torch.no_grad()
    def forward(self):
        self.q_transformer.eval()

        for episode in range(self.num_episodes):
            print(f'episode {episode}')

            instruction, curr_state = self.environment.init()

            for step in tqdm(range(self.max_num_steps_per_episode)):
                last_step = step == (self.max_num_steps_per_episode - 1)

                epsilon = self.get_epsilon(step)

                actions = self.q_transformer.get_actions(
                    rearrange(curr_state, '... -> 1 ...'),
                    [instruction],
                    prob_random_action = epsilon
                )

                reward, next_state, done = self.environment(actions)

                done = done | last_step

                # store memories using memmap, for later reflection and learning

                self.states[episode, step] = curr_state
                self.actions[episode, step] = actions
                self.rewards[episode, step] = reward
                self.dones[episode, step] = done

                # if done, move onto next episode

                if done:
                    break

                # set next state

                curr_state = next_state

            self.states.flush()
            self.actions.flush()
            self.rewards.flush()
            self.dones.flush()

        print(f'completed, memories stored to {str(self.memories_dataset_folder)}')
