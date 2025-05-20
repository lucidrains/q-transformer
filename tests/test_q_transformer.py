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
            dim_conv_stem = 64,
            dim = 64,
            dim_head = 64,
            depth = (2, 2, 5, 2),
            window_size = 7,
            mbconv_expansion_rate = 4,
            mbconv_shrinkage_rate = 0.25,
            dropout = 0.1
        ),
        num_actions = 8,
        depth = 1,
        heads = 8,
        dim_head = 64,
        cond_drop_prob = 0.2,
        dueling = True,
        weight_tie_action_bin_embed = False,
        num_residual_streams = num_residual_streams,
        dual_critics = dual_critics
    )

    video = torch.randn(2, 3, 6, 224, 224)

    instructions = [
        'bring me that apple sitting on the table',
        'please pass the butter'
    ]

    text_embeds = model.embed_texts(instructions)
    best_actions = model.get_actions(video, text_embeds = text_embeds)
    best_actions = model.get_optimal_actions(video, instructions, actions = best_actions[:, :1])

    q_values = model(video, instructions, actions = best_actions)

    q_learner = QLearner(
        model,
        dataset = MockReplayDataset(),
        n_step_q_learning = True,
        num_train_steps = 10000,
        learning_rate = 3e-4,
        batch_size = 1,
        use_bce_loss = use_bce_loss
    )
