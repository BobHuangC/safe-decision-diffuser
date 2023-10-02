from configs.base_diffuser_inv import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "diffuser_inv_d4rl_ema"
    config.log_dir_format = "{exp_name}/{env}/h_{horizon}-r_{returns_scale}-guidew_{condition_guidance_w}-dstep_{algo_cfg.num_timesteps}/{seed}"

    config.env = "hopper-medium-replay-v2"
    config.dataset = "d4rl"
    config.returns_condition = True
    config.cost_returns_condition = False
    config.env_ts_condition = True
    config.returns_scale = 400.0
    config.termination_penalty = -100.0

    config.max_traj_length = 1000
    config.horizon = 20

    config.n_epochs = 1000
    config.n_train_step_per_epoch = 1000

    config.save_period = 0

    return config
