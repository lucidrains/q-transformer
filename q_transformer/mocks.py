from random import randrange

import torch
from torch.utils.data import Dataset

class MockReplayDataset(Dataset):
    def __init__(
        self,
        length = 10000,
        num_action_bins = 256,
        video_shape = (6, 224, 224)
    ):
        self.length = length
        self.num_action_bins = num_action_bins
        self.video_shape = video_shape

    def __len__(self):
        return self.length

    def __getitem__(self, _):

        instruction = "please clean the kitchen"
        state = torch.randn(3, *self.video_shape)
        action = torch.tensor(randrange(self.num_action_bins + 1))
        next_state = torch.randn(3, *self.video_shape)
        reward = torch.tensor(randrange(2))
        done = torch.tensor(randrange(2), dtype = torch.bool)

        return instruction, state, action, next_state, reward, done

class MockReplayNStepDataset(Dataset):
    def __init__(
        self,
        length = 10000,
        num_steps = 2,
        num_action_bins = 256,
        video_shape = (6, 224, 224)
    ):
        self.num_steps = num_steps
        self.time_shape = (num_steps,)
        self.length = length
        self.num_action_bins = num_action_bins
        self.video_shape = video_shape

    def __len__(self):
        return self.length

    def __getitem__(self, _):

        instruction = "please clean the kitchen"
        state = torch.randn(*self.time_shape, 3, *self.video_shape)
        action = torch.randint(0, self.num_action_bins + 1, self.time_shape)
        next_state = torch.randn(3, *self.video_shape)
        reward = torch.randint(0, 2, self.time_shape)
        done = torch.zeros(self.time_shape, dtype = torch.bool)

        return instruction, state, action, next_state, reward, done
