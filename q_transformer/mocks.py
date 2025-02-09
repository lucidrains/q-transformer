from __future__ import annotations

from random import randrange

import torch
from torch.utils.data import Dataset

from q_transformer.tensor_typing import Float, Int, Bool
from q_transformer.agent import BaseEnvironment

class MockEnvironment(BaseEnvironment):
    def init(self) -> tuple[
        str | None,
        Float['...']
    ]:
        return 'please clean the kitchen', torch.randn(self.state_shape, device = self.device)

    def forward(self, actions) -> tuple[
        Float[''],
        Float['...'],
        Bool['']
    ]:
        rewards = torch.randn((), device = self.device)
        next_states = torch.randn(self.state_shape, device = self.device)
        done = torch.zeros((), device = self.device, dtype = torch.bool)

        return rewards, next_states, done

class MockReplayDataset(Dataset):
    def __init__(
        self,
        length = 10000,
        num_actions = 1,
        num_action_bins = 256,
        video_shape = (6, 224, 224)
    ):
        self.length = length
        self.num_actions = num_actions
        self.num_action_bins = num_action_bins
        self.video_shape = video_shape

    def __len__(self):
        return self.length

    def __getitem__(self, _):

        instruction = "please clean the kitchen"
        state = torch.randn(3, *self.video_shape)

        if self.num_actions == 1:
            action = torch.tensor(randrange(self.num_action_bins + 1))
        else:
            action = torch.randint(0, self.num_action_bins + 1, (self.num_actions,))

        next_state = torch.randn(3, *self.video_shape)
        reward = torch.tensor(randrange(2))
        done = torch.tensor(randrange(2), dtype = torch.bool)

        return instruction, state, action, next_state, reward, done

class MockReplayNStepDataset(Dataset):
    def __init__(
        self,
        length = 10000,
        num_steps = 2,
        num_actions = 1,
        num_action_bins = 256,
        video_shape = (6, 224, 224)
    ):
        self.num_steps = num_steps
        self.time_shape = (num_steps,)
        self.length = length
        self.num_actions = num_actions
        self.num_action_bins = num_action_bins
        self.video_shape = video_shape

    def __len__(self):
        return self.length

    def __getitem__(self, _):

        action_dims = (self.num_actions,) if self.num_actions > 1 else tuple()

        instruction = "please clean the kitchen"
        state = torch.randn(*self.time_shape, 3, *self.video_shape)
        action = torch.randint(0, self.num_action_bins + 1, (*self.time_shape, *action_dims))
        next_state = torch.randn(3, *self.video_shape)
        reward = torch.randint(0, 2, self.time_shape)
        done = torch.zeros(self.time_shape, dtype = torch.bool)

        return instruction, state, action, next_state, reward, done
