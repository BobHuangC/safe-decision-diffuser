from config.base_dql import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "dql_d4rl"
    config.log_dir_format = (
        "{exp_name}/{env}/lr_{algo_cfg.lr}-temp_{algo_cfg.sample_temperature}/{seed}"
    )

    config.env = "hopper-medium-expert-v2"
    config.dataset = "d4rl"
    config.max_traj_length = 1000

    config.n_epochs = 1000
    config.n_train_step_per_epoch = 1000

    config.save_period = 0

    return config
