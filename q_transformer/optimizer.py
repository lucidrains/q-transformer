from adam_atan2_pytorch import AdamAtan2

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params

def get_adam_optimizer(
    params,
    lr = 1e-4,
    wd = 0,
    betas = (0.9, 0.99),
    regen_reg_rate = 1e-2,
    filter_by_requires_grad = False,
    group_wd_params = True
):
    has_wd = wd > 0

    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if group_wd_params and has_wd:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    return AdamAtan2(params, lr = lr, weight_decay = wd, betas = betas, regen_reg_rate = regen_reg_rate)
