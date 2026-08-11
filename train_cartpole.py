# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "q-transformer",
#     "gymnasium",
#     "fire",
#     "env_ssl_wrapper",
# ]
# ///

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from shutil import rmtree

import fire
import torch
import numpy as np
import gymnasium as gym

from torch import nn
from einops import rearrange
from PIL import Image
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

from env_ssl_wrapper import compose_env
from env_ssl_wrapper.image_wrapper import ImageObservationWrapper

from q_transformer import QRoboticTransformer, QLearner
from q_transformer.agent import ReplayMemoryDataset

# constants

STATES_FILENAME = 'states.memmap.npy'
ACTIONS_FILENAME = 'actions.memmap.npy'
REWARDS_FILENAME = 'rewards.memmap.npy'
DONES_FILENAME = 'dones.memmap.npy'
device = torch.device('cpu')

# helpers

def exists(v):
    return v is not None

# encode the state vector (x, x_dot, theta, theta_dot) as a tiny image, one row each

def obs_to_pseudo_image(obs):
    ranges = [(-2.4, 2.4), (-10., 10.), (-0.21, 0.21), (-5., 5.)]
    img = np.zeros((14, 14), dtype = 'float32')

    for i, (lo, hi) in enumerate(ranges):
        img[i] = (obs[i] - lo) / (hi - lo)

    return torch.from_numpy(img.clip(0., 1.))[None, None]

def obs_to_image(obs):
    return obs['image']

def stack_frames(frames, n_dims):
    frames = torch.stack(list(frames))
    diffs = frames.diff(dim = 0)
    diffs = torch.cat((torch.zeros_like(frames[:1]), diffs), dim = 0)

    frames = rearrange(frames, f'f {n_dims} -> c f h w')
    diffs = rearrange(diffs, f'f {n_dims} -> c f h w')

    return torch.cat((frames, diffs), dim = 0)

def evaluate(
    q_transformer,
    env,
    num_frames,
    max_num_steps_per_episode,
    obs_to_input,
    n_dims,
    seed = 0
):
    q_transformer.eval()
    eval_rewards = []

    for episode_idx in range(3):
        obs, info = env.reset(seed = (seed + episode_idx) if seed is not None else None)
        frames = deque([obs_to_input(obs)] * num_frames, maxlen = num_frames)
        cumulative_reward = 0.

        for step in range(max_num_steps_per_episode):
            curr_state = stack_frames(frames, n_dims).to(device)

            with torch.no_grad():
                actions = q_transformer.get_optimal_actions(rearrange(curr_state, '... -> 1 ...'))

            obs, reward, terminated, truncated, info = env.step(int(actions[0] > 0))

            cumulative_reward += reward
            frames.append(obs_to_input(obs))

            if terminated or truncated:
                break

        eval_rewards.append(cumulative_reward)

    print(f'eval rewards: {[round(r, 1) for r in eval_rewards]}')
    return eval_rewards

# center crop the rendered frame before resizing, so the pole stays visible at small image sizes

class CroppedImageObservationWrapper(ImageObservationWrapper):
    def __init__(
        self,
        env,
        crop_width_fraction = 0.4,
        crop_height_fraction = 0.6,
        **kwargs
    ):
        super().__init__(env, **kwargs)
        self.crop_width_fraction = crop_width_fraction
        self.crop_height_fraction = crop_height_fraction

    def render_frame(self):
        img = self.env.render()

        height, width, _ = img.shape
        left = int(width * (1 - self.crop_width_fraction) / 2)
        right = int(width * (1 + self.crop_width_fraction) / 2)
        top = int(height * (1 - self.crop_height_fraction) / 2)
        bottom = int(height * (1 + self.crop_height_fraction) / 2)

        img = Image.fromarray(img).crop((left, top, right, bottom))
        img = img.resize(self.image_size, resample = self.resample_method)

        img_tensor = torch.from_numpy(np.array(img))
        img_tensor = rearrange(img_tensor, 'h w c -> 1 c h w')

        if self.normalize:
            img_tensor = img_tensor.float() / self.normalize_divisor

        return img_tensor

# contrived, tiny vision encoder used in place of the maxvit
# exposes the same interface the q-transformer expects of its `vit`

class TinyConvEncoder(nn.Module):
    def __init__(self, channels = 1, dim = 32):
        super().__init__()
        self.cond_hidden_dims = ()
        self.embed_dim = dim
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 4, stride = 2, padding = 1),
            nn.GELU(),
            nn.Conv2d(16, dim, 4, stride = 2, padding = 1),
            nn.GELU(),
        )

    def forward(self, img, *args, **kwargs):
        return self.net(img)

# dataset that stacks the last `num_frames` frames into the state

class FrameStackedReplayDataset(ReplayMemoryDataset):
    def __init__(self, folder, num_frames = 2, n_dims = 'c h w'):
        super().__init__(folder = folder, num_timesteps = 1, condition_on_text = False)
        self.num_frames = num_frames
        self.n_dims = n_dims

    def get_state_window(self, episode_index, timestep_index):
        start = max(0, timestep_index - self.num_frames + 1)
        frames = self.states[episode_index, start:(timestep_index + 1)].copy()

        if frames.shape[0] < self.num_frames:
            pad = np.repeat(frames[:1], self.num_frames - frames.shape[0], axis = 0)
            frames = np.concatenate((pad, frames), axis = 0)

        # same frame-to-frame difference channel as the rollout, so train and inference match

        diffs = np.diff(frames, axis = 0)
        diffs = np.concatenate((np.zeros_like(frames[:1]), diffs), axis = 0)

        frames = rearrange(frames, f'k {self.n_dims} -> c k h w')
        diffs = rearrange(diffs, f'k {self.n_dims} -> c k h w')

        return np.concatenate((frames, diffs), axis = 0)

    def __getitem__(self, idx):
        episode_index, timestep_index = self.indices[idx]

        states = self.get_state_window(episode_index, timestep_index)
        actions = self.actions[episode_index, timestep_index]
        rewards = self.rewards[episode_index, timestep_index]
        dones = self.dones[episode_index, timestep_index]

        next_state_timestep = min(timestep_index + 1, self.max_episode_len - 1)
        next_state = self.get_state_window(episode_index, next_state_timestep)

        # single action environment -> flat tensors, as expected by the classic q-learning path
        return None, states, int(actions), next_state, None, float(rewards), bool(dones)

def collate_fn(batch):
    _, states, actions, next_states, _, rewards, dones = zip(*batch)

    return (
        None,
        torch.from_numpy(np.stack(states)),
        torch.tensor(actions),
        torch.from_numpy(np.stack(next_states)),
        None,
        torch.tensor(rewards),
        torch.tensor(dones),
    )

# main

def main(
    obs_mode = 'pseudo',  # 'pseudo' for state-vector pseudo images, 'image' for rendered frames
    num_episodes = 120,
    max_num_steps_per_episode = 200,
    num_frames = 2,
    batch_size = 32,
    learning_rate = 3e-4,
    reward_scale = 0.01,
    gamma = 0.99,
    memories_folder = './replay_memories_data',
    checkpoint_folder = './checkpoints',
    rollout_per_loop = 40,
    train_per_loop = 250,
    epsilon_start = 0.5,
    epsilon_end = 0.15,
    num_steps_to_target_epsilon = 800,
    solve_threshold = 100,
    seed = 0,
):
    print(f'starting cartpole q-transformer training (obs mode: {obs_mode})...')

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    mem_path = Path(memories_folder)
    rmtree(mem_path, ignore_errors = True)
    mem_path.mkdir(exist_ok = True, parents = True)

    if obs_mode == 'image':
        obs_to_input = obs_to_image
        n_dims = 'c h w'
        channels = 2 * 3
        state_shape = (3, 64, 64)
        env = compose_env(
            gym.make('CartPole-v1', render_mode = 'rgb_array'),
            (CroppedImageObservationWrapper, dict(image_size = (64, 64))),
        )
    else:
        obs_to_input = obs_to_pseudo_image
        n_dims = 'c 1 h w'
        channels = 2
        state_shape = (1, 1, 14, 14)
        env = gym.make('CartPole-v1')

    # model: tiny conv encoder + q transformer, single discrete action with 2 bins
    # each state is num_frames frames plus their frame-to-frame differences

    q_transformer = QRoboticTransformer(
        vit = TinyConvEncoder(channels = channels, dim = 32),
        num_actions = 1,
        action_bins = 2,
        depth = 2,
        heads = 4,
        dim_head = 32,
        condition_on_text = False,
        dueling = False,
        weight_tie_action_bin_embed = False,
    ).to(device)

    num_params = sum(p.numel() for p in q_transformer.parameters())
    print(f'model parameters: {num_params / 1e6:.2f}M')

    # memmap storage (single frames; stacking happens in the dataset)

    prec_shape = (num_episodes, max_num_steps_per_episode)

    states_mm  = open_memmap(str(mem_path / STATES_FILENAME), dtype = 'float32', mode = 'w+', shape = (*prec_shape, *state_shape))
    actions_mm = open_memmap(str(mem_path / ACTIONS_FILENAME), dtype = 'int',     mode = 'w+', shape = (*prec_shape, 1))
    rewards_mm = open_memmap(str(mem_path / REWARDS_FILENAME), dtype = 'float32', mode = 'w+', shape = prec_shape)
    dones_mm   = open_memmap(str(mem_path / DONES_FILENAME),   dtype = 'bool',    mode = 'w+', shape = prec_shape)

    # epsilon decays over global environment steps

    epsilon_slope = (epsilon_end - epsilon_start) / num_steps_to_target_epsilon
    env_step = 0

    num_loops = num_episodes // rollout_per_loop

    ql_learner = None

    for loop_i in range(num_loops):
        print(f'\n--- Loop {loop_i + 1}/{num_loops} ---')

        # 1. rollout with epsilon greedy

        q_transformer.eval()

        for ep_offset in range(rollout_per_loop):
            episode_idx = loop_i * rollout_per_loop + ep_offset

            obs, info = env.reset(seed = (seed + episode_idx) if seed is not None else None)
            frames = deque([obs_to_input(obs)] * num_frames, maxlen = num_frames)

            cumulative_reward = 0.

            for step in range(max_num_steps_per_episode):
                last_step = step == (max_num_steps_per_episode - 1)

                epsilon = max(epsilon_end, epsilon_slope * env_step + epsilon_start)
                env_step += 1

                curr_frame = frames[-1]
                curr_state = stack_frames(frames, n_dims).to(device)

                with torch.no_grad():
                    actions = q_transformer.get_actions(
                        rearrange(curr_state, '... -> 1 ...'),
                        prob_random_action = epsilon,
                    )

                env_action = int(actions[0] > 0)  # bin 1 -> push right, bin 0 -> push left

                obs, reward, terminated, truncated, info = env.step(env_action)

                done = terminated or truncated or last_step
                cumulative_reward += reward

                states_mm[episode_idx, step]  = curr_frame.numpy()
                actions_mm[episode_idx, step] = actions.cpu().numpy()
                rewards_mm[episode_idx, step] = reward * reward_scale
                dones_mm[episode_idx, step]   = done

                frames.append(obs_to_input(obs))

                if done:
                    break

            for f in (states_mm, actions_mm, rewards_mm, dones_mm):
                f.flush()

            print(f'ep {episode_idx}: reward {cumulative_reward:.1f}, epsilon {epsilon:.2f}')

        # 2. q-learning on all memories collected so far

        dataset = FrameStackedReplayDataset(memories_folder, num_frames = num_frames, n_dims = n_dims)

        if not exists(ql_learner):
            ql_learner = QLearner(
                q_transformer,
                dataset = dataset,
                batch_size = batch_size,
                num_train_steps = train_per_loop,
                learning_rate = learning_rate,
                discount_factor_gamma = gamma,
                conservative_reg_loss_weight = 0.,
                q_target_ema_kwargs = dict(beta = 0.99, update_after_step = 0, update_every = 1),
                accelerator_kwargs = dict(cpu = True),
                checkpoint_folder = checkpoint_folder,
                dataloader_kwargs = dict(shuffle = True, collate_fn = collate_fn),
            )
        else:
            ql_learner.dataloader = DataLoader(
                dataset,
                batch_size = batch_size,
                shuffle = True,
                collate_fn = collate_fn,
            )
            ql_learner.num_train_steps += train_per_loop

        ql_learner.forward()

        # 3. greedy evaluation of the current policy

        eval_rewards = evaluate(
            q_transformer,
            env,
            num_frames,
            max_num_steps_per_episode,
            obs_to_input,
            n_dims,
            seed = seed + loop_i * 5,
        )

        mean_eval_reward = sum(eval_rewards) / len(eval_rewards)

        if mean_eval_reward >= solve_threshold:
            print(f'solved! stopping early after loop {loop_i + 1}')
            break

    print('training completed.')
if __name__ == '__main__':
    fire.Fire(main)
