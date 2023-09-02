from configs.base_diffuser_inv import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "diffuser_inv_dsrl"
    config.log_dir_format = "{exp_name}/{env}/h_{horizon}-tgt_{target_returns}-usecost_{cost_returns_condition}-guidew_{condition_guidance_w}/{seed}"

    config.env = "OfflineBallRun-v0"
    config.dataset = "dsrl"
    config.returns_condition = True
    config.cost_returns_condition = True
    config.target_returns = "500.0,10.0"

    config.max_traj_length = 100
    config.horizon = 20

    # data aug configs
    config.aug_percent = 0.3
    config.aug_deg = 2
    config.aug_max_rew_decrease = 100.0
    config.aug_beta = 1.0
    config.aug_max_reward = 500.0
    config.aug_min_reward = 1.0

    config.n_epochs = 500
    config.n_train_step_per_epoch = 1000

    config.save_period = 0

    return config
