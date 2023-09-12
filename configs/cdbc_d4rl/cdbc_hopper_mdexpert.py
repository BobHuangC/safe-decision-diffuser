from configs.base_cdbc import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "cdbc_d4rl"
    config.log_dir_format = (
        "{exp_name}/{env}/tgt_{target_returns}-guidew_{condition_guidance_w}/{seed}"
    )

    config.env = "hopper-medium-expert-v2"
    config.dataset = "d4rl"
    config.returns_condition = True
    config.cost_returns_condition = False
    config.env_ts_condition = True
    config.max_traj_length = 1000

    config.target_returns = "3500.0,0.0"
    config.termination_penalty = -100.0

    config.n_epochs = 1000
    config.n_train_step_per_epoch = 1000

    config.save_period = 0

    return config
