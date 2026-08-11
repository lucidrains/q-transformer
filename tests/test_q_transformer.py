import pytest
import torch

from q_transformer import (
    QRoboticTransformer,
    QLearner,
    ReplayMemoryDataset
)

from q_transformer.mocks import (
    MockReplayDataset,
    MockReplayNStepDataset
)


@pytest.mark.parametrize('num_residual_streams', (1, 4))
@pytest.mark.parametrize('use_bce_loss', (False, True))
@pytest.mark.parametrize('dual_critics', (False, True))
def test_q_transformer(
    num_residual_streams,
    use_bce_loss,
    dual_critics
):

    model = QRoboticTransformer(
        vit = dict(
            num_classes = 1000,
            dim_conv_stem = 32,
            dim = 32,
            dim_head = 32,
            depth = (1, 1, 1, 1),
            window_size = 1,
            mbconv_expansion_rate = 2,
            mbconv_shrinkage_rate = 0.25,
            dropout = 0.1
        ),
        num_actions = 8,
        depth = 1,
        heads = 4,
        dim_head = 32,
        cond_drop_prob = 0.2,
        dueling = True,
        weight_tie_action_bin_embed = False,
        num_residual_streams = num_residual_streams,
        dual_critics = dual_critics
    )

    video = torch.randn(2, 3, 6, 32, 32)

    instructions = [
        'bring me that apple sitting on the table',
        'please pass the butter'
    ]

    text_embeds = model.embed_texts(instructions)
    best_actions = model.get_actions(video, text_embeds = text_embeds)
    best_actions = model.get_optimal_actions(video, text_embeds = text_embeds, actions = best_actions[:, :1])

    q_values = model(video, text_embeds = text_embeds, actions = best_actions)

    q_learner = QLearner(
        model,
        dataset = MockReplayDataset(video_shape = (6, 32, 32)),
        n_step_q_learning = True,
        num_train_steps = 2,
        learning_rate = 3e-4,
        batch_size = 1,
        use_bce_loss = use_bce_loss
    )
