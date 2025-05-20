from __future__ import annotations

from pathlib import Path
from functools import partial
from contextlib import nullcontext
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype

from q_transformer.q_robotic_transformer import QRoboticTransformer
from q_transformer.agent import ReplayMemoryDataset

from adam_atan2_pytorch import AdamAtan2

from hl_gauss_pytorch import BinaryHLGaussLoss

from q_transformer.tensor_typing import (
    Float,
    Int,
    Bool
)

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from ema_pytorch import EMA

# constants

QIntermediates = namedtuple('QIntermediates', [
    'q_pred_all_actions',
    'q_pred',
    'q_next',
    'q_target'
])

Losses = namedtuple('Losses', [
    'td_loss',
    'conservative_reg_loss'
])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_divisible(num, den):
    return (num % den) == 0

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack_one(x, pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack_one(out, packed_shape, inv_pattern)

    return packed, inverse

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# tensor helpers

def batch_select_indices(t, indices):
    indices = rearrange(indices, '... -> ... 1')
    selected = t.gather(-1, indices)
    return rearrange(selected, '... 1 -> ...')

# Q learning on robotic transformer

class QLearner(Module):

    @beartype
    def __init__(
        self,
        model: QRoboticTransformer | Module,
        *,
        dataset: Dataset,
        batch_size: int,
        num_train_steps: int,
        learning_rate: float,
        min_reward: float = 0.,
        grad_accum_every: int = 1,
        monte_carlo_return: float | None = None,
        use_bce_loss: bool = False,   # if set to True, will use a classification loss instead of regression (replacing all mse)
        hl_gauss_sigma: float = 0.15, # amount of label smoothing of the two bin, but may increase this to more bins if see a signal
        weight_decay: float = 0.,
        regen_reg_rate: float = 1e-3,
        accelerator: Accelerator | None = None,
        accelerator_kwargs: dict = dict(),
        dataloader_kwargs: dict = dict(
            shuffle = True
        ),
        q_target_ema_kwargs: dict = dict(
            beta = 0.99,
            update_after_step = 10,
            update_every = 5
        ),
        max_grad_norm = 0.5,
        n_step_q_learning = False,
        discount_factor_gamma = 0.98,
        conservative_reg_loss_weight = 1., # they claim 1. is best in paper
        checkpoint_folder = './checkpoints',
        checkpoint_every = 1000,
    ):
        super().__init__()

        self.is_multiple_actions = model.num_actions > 1

        # q-learning related hyperparameters

        self.discount_factor_gamma = discount_factor_gamma
        self.n_step_q_learning = n_step_q_learning

        self.has_conservative_reg_loss = conservative_reg_loss_weight > 0.
        self.conservative_reg_loss_weight = conservative_reg_loss_weight

        self.register_buffer('discount_matrix', None, persistent = False)

        # online (evaluated) Q model

        self.model = model

        # ema (target) Q model

        self.ema_model = EMA(
            model,
            include_online_model = False,
            **q_target_ema_kwargs
        )

        self.max_grad_norm = max_grad_norm

        self.optimizer = AdamAtan2(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay,
            regen_reg_rate = regen_reg_rate,
        )

        if not exists(accelerator):
            accelerator = Accelerator(
                kwargs_handlers = [
                    DistributedDataParallelKwargs(find_unused_parameters = True)
                ],
                **accelerator_kwargs
            )

        self.accelerator = accelerator

        self.min_reward = min_reward

        # use classification loss, from "Stop Regressing" paper Farebrother et al.
        # histogram loss from Imani et al.

        self.use_bce_loss = use_bce_loss

        self.hl_gauss_loss = BinaryHLGaussLoss(
            sigma = hl_gauss_sigma,
        )

        self.hl_gauss_loss.to(accelerator.device)

        self.monte_carlo_return = monte_carlo_return

        # dataset from replay memories

        if isinstance(dataset, ReplayMemoryDataset):
            assert dataset.num_actions == model.num_actions, f'the ReplayMemoryDataset is loading memories where the transformer had {dataset.num_actions} actions but the model has only {model.num_actions} actions'

        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            **dataloader_kwargs
        )

        # prepare

        (
            self.model,
            self.ema_model,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.ema_model,
            self.optimizer,
            self.dataloader
        )

        # checkpointing related

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)

        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        # dummy loss

        self.register_buffer('zero', torch.tensor(0.))

        # training step related

        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every

        self.register_buffer('step', torch.tensor(0))

    def save(
        self,
        checkpoint_num = None,
        overwrite = True
    ):
        name = 'checkpoint'
        if exists(checkpoint_num):
            name += f'-{checkpoint_num}'

        path = self.checkpoint_folder / (name + '.pt')

        assert overwrite or not path.exists()

        pkg = dict(
            model = self.unwrap(self.model).state_dict(),
            ema_model = self.unwrap(self.ema_model).state_dict(),
            optimizer = self.optimizer.state_dict(),
            step = self.step.item()
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)
        assert exists(path)

        pkg = torch.load(str(path), weights_only = True)

        self.unwrap(self.model).load_state_dict(pkg['model'])
        self.unwrap(self.ema_model).load_state_dict(pkg['ema_model'])

        self.optimizer.load_state_dict(pkg['optimizer'])
        self.step.copy_(pkg['step'])

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def unwrap(self, module):
        return self.accelerator.unwrap_model(module)

    def print(self, msg):
        return self.accelerator.print(msg)

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def get_discount_matrix(self, timestep):
        if exists(self.discount_matrix) and self.discount_matrix.shape[-1] >= timestep:
            return self.discount_matrix[:timestep, :timestep]

        timestep_arange = torch.arange(timestep, device = self.accelerator.device)
        powers = (timestep_arange[None, :] - timestep_arange[:, None])
        discount_matrix = torch.triu(self.discount_factor_gamma ** powers)

        self.register_buffer('discount_matrix', discount_matrix, persistent = False)
        return self.discount_matrix

    def q_learn(
        self,
        text_embeds:    Float['b d'],
        states:         Float['b c f h w'],
        actions:        Int[' b'],
        next_states:    Float['b c f h w'],
        reward:         Float[' b'],
        done:           Bool[' b'],
        *,
        monte_carlo_return = None

    ) -> tuple[Float[''], QIntermediates]:
        # 'next' stands for the very next time step (whether state, q, actions etc)

        γ = self.discount_factor_gamma
        not_terminal = (~done).float()

        # first make a prediction with online q robotic transformer
        # select out the q-values for the action that was taken

        q_pred_all_actions = self.model(states, text_embeds = text_embeds)
        q_pred = batch_select_indices(q_pred_all_actions, actions)

        # use an exponentially smoothed copy of model for the future q target. more stable than setting q_target to q_eval after each batch
        # the max Q value is taken as the optimal action is implicitly the one with the highest Q score

        q_next = self.ema_model(next_states, text_embeds = text_embeds).amax(dim = -1)
        q_next.clamp_(min = default(monte_carlo_return, -1e4))

        # get the q_next_value, which depends on whether classification loss being used

        if self.use_bce_loss:
            q_next_value = self.hl_gauss_loss.transform_from_logits(q_next)
        else:
            q_next_value = q_next.sigmoid()

        # Bellman's equation. most important line of code, hopefully done correctly

        q_target = reward + not_terminal * (γ * q_next_value)

        # now just force the online model to be able to predict this target

        if self.use_bce_loss:
            loss = self.hl_gauss_loss(q_pred, q_target)
        else:
            loss = F.mse_loss(q_pred.sigmoid(), q_target)

        # that's it. ~5 loc for the heart of q-learning
        # return loss and some of the intermediates for logging

        return loss, QIntermediates(q_pred_all_actions, q_pred, q_next, q_target)

    def n_step_q_learn(
        self,
        text_embeds:    Float['b t d'],
        states:         Float['b t c f h w'],
        actions:        Int['b t'],
        next_states:    Float['b c f h w'],
        next_text_embed: Float['b d'],
        rewards:        Float['b t'],
        dones:          Bool['b t'],
        *,
        monte_carlo_return = None

    ) -> Tuple[Float[""], QIntermediates]:
        """
        einops

        b - batch
        c - channels
        f - frames
        h - height
        w - width
        t - timesteps
        a - action bins
        q - q values
        d - text cond dimension
        """

        num_timesteps, device = states.shape[1], states.device

        # fold time steps into batch

        states, inv_pack_time = pack_one_with_inverse(states, '* c f h w')
        text_embeds, _ = pack_one(text_embeds, '* d')

        # repeat text embeds per timestep

        repeated_text_embeds = repeat(text_embeds, 'b ... -> (b n) ...', n = num_timesteps)

        γ = self.discount_factor_gamma

        # anything after the first done flag will be considered terminal

        dones = dones.cumsum(dim = -1) > 0
        dones = F.pad(dones, (1, 0), value = False)

        not_terminal = (~dones).float()

        # get q predictions

        actions = rearrange(actions, 'b t -> (b t)')

        q_pred_all_actions = self.model(states, text_embeds = repeated_text_embeds)
        q_pred = batch_select_indices(q_pred_all_actions, actions)
        q_pred = inv_pack_time(q_pred, '*')

        q_next = self.ema_model(next_states, text_embeds = next_text_embed).amax(dim = -1)
        q_next.clamp_(min = default(monte_carlo_return, -1e4))

        # determine the q_next_value

        if self.use_bce_loss:
            q_next_value = self.hl_gauss_loss.transform_from_logits(q_next)
        else:
            q_next_value = q_next.sigmoid()

        # prepare rewards and discount factors across timesteps

        rewards, _ = pack([rewards, q_next_value], 'b *')

        γ = self.get_discount_matrix(num_timesteps + 1)[:-1, :]

        # account for discounting using the discount matrix

        q_target = einsum('b t, q t -> b q', not_terminal * rewards, γ)

        # have transformer learn to predict above Q target

        if self.use_bce_loss:
            loss = self.hl_gauss_loss(q_pred, q_target)
        else:
            loss = F.mse_loss(q_pred.sigmoid(), q_target)

        # prepare q prediction

        q_pred_all_actions = inv_pack_time(q_pred_all_actions, '* a')

        return loss, QIntermediates(q_pred_all_actions, q_pred, q_next, q_target)

    def autoregressive_q_learn_handle_single_timestep(
        self,
        text_embeds,
        states,
        actions,
        next_states,
        next_text_embed,
        rewards,
        dones,
        *,
        monte_carlo_return = None
    ):
        """
        simply detect and handle single timestep
        and use `autoregressive_q_learn` as more general function
        """
        if states.ndim == 5:
            states = rearrange(states, 'b ... -> b 1 ...')

        if actions.ndim == 2:
            actions = rearrange(actions, 'b ... -> b 1 ...')

        if rewards.ndim == 1:
            rewards = rearrange(rewards, 'b -> b 1')

        if dones.ndim == 1:
            dones = rearrange(dones, 'b -> b 1')

        return self.autoregressive_q_learn(text_embeds, states, actions, next_states, next_text_embed, rewards, dones, monte_carlo_return = monte_carlo_return)

    def autoregressive_q_learn(
        self,
        text_embeds:     Float['b t d'],
        states:          Float['b t c f h w'],
        actions:         Int['b t n'],
        next_states:     Float['b c f h w'],
        next_text_embed: Float['b d'],
        rewards:         Float['b t'],
        dones:           Bool['b t'],
        *,
        monte_carlo_return = None

    ) -> tuple[Float[''], QIntermediates]:
        """
        einops

        b - batch
        c - channels
        f - frames
        h - height
        w - width
        t - timesteps
        n - number of actions
        a - action bins
        q - q values
        d - text cond dimension
        """
        monte_carlo_return = default(monte_carlo_return, -1e4)
        num_timesteps, device = states.shape[1], states.device

        # fold time steps into batch

        states, inv_pack_time = pack_one_with_inverse(states, '* c f h w')
        actions, _ = pack_one(actions, '* n')
        text_embeds, _ = pack_one(text_embeds, '* d')

        # anything after the first done flag will be considered terminal

        dones = dones.cumsum(dim = -1) > 0
        dones = F.pad(dones, (1, -1), value = False)

        not_terminal = (~dones).float()

        # rewards should not be given on and after terminal step

        rewards = rewards * not_terminal

        # because greek unicode is nice to look at

        γ = self.discount_factor_gamma

        # get predicted Q for each action
        # unpack back to (b, t, n)

        q_pred_all_actions = self.model(states, text_embeds = text_embeds, actions = actions)
        q_pred = batch_select_indices(q_pred_all_actions, actions)
        q_pred = inv_pack_time(q_pred, '* n')

        # get q_next

        q_next = self.ema_model(next_states, text_embeds = next_text_embed)
        q_next = q_next.max(dim = -1).values
        q_next.clamp_(min = monte_carlo_return)

        # get target Q
        # unpack back to - (b, t, n)

        q_target_all_actions = self.ema_model(states, text_embeds = text_embeds, actions = actions)
        q_target = q_target_all_actions.max(dim = -1).values

        q_target.clamp_(min = monte_carlo_return)
        q_target = inv_pack_time(q_target, '* n')

        # main contribution of the paper is the following logic
        # section 4.1 - eq. 1

        # first take care of the loss for all actions except for the very last one

        q_pred_rest_actions, q_pred_last_action      = q_pred[..., :-1], q_pred[..., -1]
        q_target_first_action, q_target_rest_actions = q_target[..., 0], q_target[..., 1:]

        if self.use_bce_loss:
            q_target_rest_actions_values = self.hl_gauss_loss.transform_from_logits(q_target_rest_actions)
            losses_all_actions_but_last = self.hl_gauss_loss(q_pred_rest_actions, q_target_rest_actions_values, reduction = 'none')
        else:
            losses_all_actions_but_last = F.mse_loss(q_pred_rest_actions.sigmoid(), q_target_rest_actions.sigmoid(), reduction = 'none')

        # next take care of the very last action, which incorporates the rewards

        q_target_last_action, _ = pack([q_target_first_action[..., 1:], q_next], 'b *')

        if self.use_bce_loss:
            q_target_last_action_value = self.hl_gauss_loss.transform_from_logits(q_target_last_action)
        else:
            q_target_last_action_value = q_target_last_action.sigmoid()

        # Bellman's equation

        q_target_last_action_value = rewards + γ * q_target_last_action_value

        # loss

        if self.use_bce_loss:
            losses_last_action = self.hl_gauss_loss(q_pred_last_action, q_target_last_action_value, reduction = 'none')
        else:
            losses_last_action = F.mse_loss(q_pred_last_action.sigmoid(), q_target_last_action, reduction = 'none')

        # flatten and average

        losses, _ = pack([losses_all_actions_but_last, losses_last_action], '*')

        return losses.mean(), QIntermediates(q_pred_all_actions, q_pred, q_next, q_target)

    def learn(
        self,
        *args,
        min_reward: float | None = None,
        monte_carlo_return: float | None = None
    ):
        _, _, actions, *_ = args

        # q-learn kwargs

        q_learn_kwargs = dict(
            monte_carlo_return = monte_carlo_return
        )

        # main q-learning loss, respectively
        # 1. proposed autoregressive q-learning for multiple actions - (handles single or n-step automatically)
        # 2. single action - single timestep (classic q-learning)
        # 3. single action - n-steps

        if self.is_multiple_actions:
            td_loss, q_intermediates = self.autoregressive_q_learn_handle_single_timestep(*args, **q_learn_kwargs)
            num_timesteps = actions.shape[1]

        elif self.n_step_q_learning:
            td_loss, q_intermediates = self.n_step_q_learn(*args, **q_learn_kwargs)
            num_timesteps = actions.shape[1]

        else:
            td_loss, q_intermediates = self.q_learn(*args, **q_learn_kwargs)
            num_timesteps = 1

        if not self.has_conservative_reg_loss:
            return loss, Losses(td_loss, self.zero)

        # calculate conservative regularization
        # section 4.2 in paper, eq 2

        batch = actions.shape[0]

        q_preds = q_intermediates.q_pred_all_actions
        q_preds = rearrange(q_preds, '... a -> (...) a')

        num_action_bins = q_preds.shape[-1]
        num_non_dataset_actions = num_action_bins - 1

        actions = rearrange(actions, '... -> (...) 1')

        dataset_action_mask = torch.zeros_like(q_preds).scatter_(-1, actions, torch.ones_like(q_preds))

        q_actions_not_taken = q_preds[~dataset_action_mask.bool()]
        q_actions_not_taken = rearrange(q_actions_not_taken, '(b t a) -> b t a', b = batch, a = num_non_dataset_actions)

        conservative_reg_loss = ((q_actions_not_taken - (min_reward * num_timesteps)) ** 2).sum() / num_non_dataset_actions

        # total loss

        loss =  0.5 * td_loss + \
                0.5 * conservative_reg_loss * self.conservative_reg_loss_weight

        loss_breakdown = Losses(td_loss, conservative_reg_loss)

        return loss, loss_breakdown

    def forward(
        self,
        *,
        monte_carlo_return: float | None = None,
        min_reward: float | None = None
    ):
        monte_carlo_return = default(monte_carlo_return, self.monte_carlo_return)
        min_reward = default(min_reward, self.min_reward)

        step = self.step.item()

        replay_buffer_iter = cycle(self.dataloader)

        self.model.train()
        self.ema_model.train()

        while step < self.num_train_steps:

            # zero grads

            self.optimizer.zero_grad()

            # main q-learning algorithm

            for grad_accum_step in range(self.grad_accum_every):
                is_last = grad_accum_step == (self.grad_accum_every - 1)
                context = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                with self.accelerator.autocast(), context():

                    loss, (td_loss, conservative_reg_loss) = self.learn(
                        *next(replay_buffer_iter),
                        min_reward = min_reward,
                        monte_carlo_return = monte_carlo_return
                    )

                    self.accelerator.backward(loss / self.grad_accum_every)

            self.print(f'td loss: {td_loss.item():.3f}')

            # clip gradients (transformer best practices)

            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # take optimizer step

            self.optimizer.step()

            # update target ema

            self.wait()

            self.ema_model.update()

            # increment step

            step += 1
            self.step.add_(1)

            # whether to checkpoint or not

            self.wait()

            if self.is_main and is_divisible(step, self.checkpoint_every):
                checkpoint_num = step // self.checkpoint_every
                self.save(checkpoint_num)

            self.wait()

        self.print('training complete')
