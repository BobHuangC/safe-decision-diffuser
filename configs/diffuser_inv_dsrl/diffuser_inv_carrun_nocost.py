from configs.base_diffuser_inv import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "diffuser_inv_dsrl"
    config.log_dir_format = "{exp_name}/{env}/h_{horizon}-hh_{history_horizon}-tgt_{target_returns}-gn_{algo_cfg.max_grad_norm}-usecost_{cost_returns_condition}/{seed}"

    config.env = "OfflineCarRun-v0"
    config.dataset = "dsrl"
    config.returns_condition = True
    config.cost_returns_condition = False
    config.env_ts_condition = True
    config.target_returns = "575.0,10.0"
    config.cost_limit = 10.0

    config.max_traj_length = 200
    config.horizon = 12
    config.history_horizon = 8

    config.eval_period = 50
    config.eval_n_trajs = 20
    config.num_eval_envs = 20

    # data aug configs
    config.aug_percent = 0.0
    config.aug_deg = 0
    config.aug_max_rew_decrease = 100.0
    config.aug_beta = 1.0
    config.aug_max_reward = 600.0
    config.aug_min_reward = 1.0

    config.n_epochs = 500
    config.n_train_step_per_epoch = 1000

    config.save_period = 0
    config.algo_cfg.max_grad_norm = 4.0

    return config
