from __future__ import annotations

import sys
from pathlib import Path

from numpy.lib.format import open_memmap

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset

from einops import rearrange

from q_transformer.q_robotic_transformer import QRoboticTransformer

from q_transformer.tensor_typing import Float, Bool

from beartype import beartype
from beartype.typing import Iterator

from tqdm import tqdm

# just force training on 64 bit systems

assert sys.maxsize > (2 ** 32), 'you need to be on 64 bit system to store > 2GB experience for your q-transformer agent'

# constants

TEXT_EMBEDS_FILENAME = 'text_embeds.memmap.npy'
STATES_FILENAME = 'states.memmap.npy'
ACTIONS_FILENAME = 'actions.memmap.npy'
REWARDS_FILENAME = 'rewards.memmap.npy'
DONES_FILENAME = 'dones.memmap.npy'

DEFAULT_REPLAY_MEMORIES_FOLDER = './replay_memories_data'

# helpers

def exists(v):
    return v is not None

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

# replay memory dataset

class ReplayMemoryDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder: str = DEFAULT_REPLAY_MEMORIES_FOLDER,
        num_timesteps: int = 1
    ):
        assert num_timesteps >= 1
        self.is_single_timestep = num_timesteps == 1
        self.num_timesteps = num_timesteps

        folder = Path(folder)
        assert folder.exists() and folder.is_dir()

        text_embeds_path = folder / TEXT_EMBEDS_FILENAME
        states_path = folder / STATES_FILENAME
        actions_path = folder / ACTIONS_FILENAME
        rewards_path = folder / REWARDS_FILENAME
        dones_path = folder / DONES_FILENAME

        self.text_embeds = open_memmap(str(text_embeds_path), dtype = 'float32', mode = 'r')
        self.states = open_memmap(str(states_path), dtype = 'float32', mode = 'r')
        self.actions = open_memmap(str(actions_path), dtype = 'int', mode = 'r')
        self.rewards = open_memmap(str(rewards_path), dtype = 'float32', mode = 'r')
        self.dones = open_memmap(str(dones_path), dtype = 'bool', mode = 'r')

        self.num_timesteps = num_timesteps

        self.num_actions = self.actions.shape[-1]

        # calculate episode length based on dones
        # filter out any episodes that are insufficient in length

        self.episode_length = (self.dones.cumsum(axis = -1) == 0).sum(axis = -1) + 1

        trainable_episode_indices = self.episode_length >= num_timesteps

        self.text_embeds = self.text_embeds[trainable_episode_indices]
        self.states = self.states[trainable_episode_indices]
        self.actions = self.actions[trainable_episode_indices]
        self.rewards = self.rewards[trainable_episode_indices]
        self.dones = self.dones[trainable_episode_indices]

        self.episode_length = self.episode_length[trainable_episode_indices]

        assert self.dones.size > 0, 'no trainable episodes'

        self.num_episodes, self.max_episode_len = self.dones.shape

        timestep_arange = torch.arange(self.max_episode_len)

        timestep_indices = torch.stack(torch.meshgrid(
            torch.arange(self.num_episodes),
            timestep_arange
        ), dim = -1)

        trainable_mask = timestep_arange < rearrange(torch.from_numpy(self.episode_length) - num_timesteps, 'e -> e 1')
        self.indices = timestep_indices[trainable_mask]

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        episode_index, timestep_index = self.indices[idx]

        timestep_slice = slice(timestep_index, (timestep_index + self.num_timesteps))

        text_embeds = self.text_embeds[episode_index, timestep_slice].copy()
        states = self.states[episode_index, timestep_slice].copy()
        actions = self.actions[episode_index, timestep_slice].copy()
        rewards = self.rewards[episode_index, timestep_slice].copy()
        dones = self.dones[episode_index, timestep_slice].copy()

        next_state_timestep = min(timestep_index, self.max_episode_len - 1)
        next_state = self.states[episode_index, next_state_timestep].copy()
        next_text_embed = self.text_embeds[episode_index, next_state_timestep].copy()

        return text_embeds, states, actions, next_state, next_text_embed, rewards, dones

# base environment class to extend

class BaseEnvironment(Module):
    @beartype
    def __init__(
        self,
        *,
        state_shape: tuple[int, ...],
        text_embed_shape: int | tuple[int, ...]
    ):
        super().__init__()
        self.state_shape = state_shape
        self.text_embed_shape = cast_tuple(text_embed_shape)
        self.register_buffer('dummy', torch.zeros(0), persistent = False)

    @property
    def device(self):
        return self.dummy.device

    def init(self) -> tuple[str, Tensor]: # (instruction, initial state)
        raise NotImplementedError

    def forward(
        self,
        actions: Int['...']
    ) -> tuple[
        Float[''],     # reward
        Float['...'],  # next state
        Bool['']       # done
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
        memories_dataset_folder: str = DEFAULT_REPLAY_MEMORIES_FOLDER,
        num_episodes: int = 1000,
        max_num_steps_per_episode: int = 10000,
        epsilon_start: float = 0.25,
        epsilon_end: float = 0.001,
        num_steps_to_target_epsilon: int = 1000
    ):
        super().__init__()
        self.q_transformer = q_transformer
    
        condition_on_text = q_transformer.condition_on_text
        self.condition_on_text = condition_on_text

        self.environment = environment

        assert hasattr(environment, 'state_shape') and hasattr(environment, 'text_embed_shape')

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

        states_path = mem_path / STATES_FILENAME
        actions_path = mem_path / ACTIONS_FILENAME
        rewards_path = mem_path / REWARDS_FILENAME
        dones_path = mem_path / DONES_FILENAME

        prec_shape = (num_episodes, max_num_steps_per_episode)
        num_actions = q_transformer.num_actions
        state_shape = environment.state_shape

        if condition_on_text:
            text_embeds_path = mem_path / TEXT_EMBEDS_FILENAME
            text_embed_shape = environment.text_embed_shape
            self.text_embed_shape = text_embed_shape
            self.text_embeds = open_memmap(str(text_embeds_path), dtype = 'float32', mode = 'w+', shape = (*prec_shape, *text_embed_shape))

        self.states      = open_memmap(str(states_path), dtype = 'float32', mode = 'w+', shape = (*prec_shape, *state_shape))
        self.actions     = open_memmap(str(actions_path), dtype = 'int', mode = 'w+', shape = (*prec_shape, num_actions))
        self.rewards     = open_memmap(str(rewards_path), dtype = 'float32', mode = 'w+', shape = prec_shape)
        self.dones       = open_memmap(str(dones_path), dtype = 'bool', mode = 'w+', shape = prec_shape)

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

                text_embed = None

                if self.condition_on_text:
                    text_embed = self.q_transformer.embed_texts([instruction])

                actions = self.q_transformer.get_actions(
                    rearrange(curr_state, '... -> 1 ...'),
                    text_embeds = text_embed,
                    prob_random_action = epsilon
                )

                reward, next_state, done = self.environment(actions)

                done = done | last_step

                # store memories using memmap, for later reflection and learning

                if self.condition_on_text:
                    assert text_embed.shape[1:] == self.text_embed_shape
                    self.text_embeds[episode, step] = text_embed

                self.states[episode, step]      = curr_state
                self.actions[episode, step]     = actions
                self.rewards[episode, step]     = reward
                self.dones[episode, step]       = done

                # if done, move onto next episode

                if done:
                    break

                # set next state

                curr_state = next_state

            if self.condition_on_text:
                self.text_embeds.flush()

            self.states.flush()
            self.actions.flush()
            self.rewards.flush()
            self.dones.flush()

        # close memmap

        if self.condition_on_text:
            del self.text_embeds

        del self.states
        del self.actions
        del self.rewards
        del self.dones

        print(f'completed, memories stored to {self.memories_dataset_folder.resolve()}')
