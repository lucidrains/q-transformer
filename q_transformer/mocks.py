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
        done = torch.tensor(randrange(2))

        return instruction, state, action, next_state, reward, done
