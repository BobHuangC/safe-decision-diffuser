from configs.base_dql import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "dql_dsrl"
    config.log_dir_format = (
        "{exp_name}/{env}/lr_{algo_cfg.lr}-temp_{algo_cfg.sample_temperature}/{seed}"
    )

    config.env = "OfflineBallRun-v0"
    config.dataset = "dsrl"
    config.returns_condition = True
    config.cost_returns_condition = True
    config.env_ts_condition = True
    config.target_returns = "500.0,10.0"
    config.cost_limit = 10.0

    config.max_traj_length = 1000
    config.horizon = 1

    config.eval_period = 50
    config.eval_n_trajs = 20
    config.num_eval_envs = 20

    # data aug configs
    config.aug_percent = 0.2
    config.aug_deg = 2
    config.aug_max_rew_decrease = 200.0
    config.aug_beta = 1.0
    config.aug_max_reward = 1400.0
    config.aug_min_reward = 1.0

    config.n_epochs = 1000
    config.n_train_step_per_epoch = 1000

    config.save_period = 0

    return config
