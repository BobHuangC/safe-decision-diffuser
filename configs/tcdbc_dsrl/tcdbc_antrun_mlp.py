from configs.base_cdbc import get_base_config

def get_config():
    config = get_base_config()
    config.exp_name = "tcdbc_dsrl"
    config.log_dir_format = (
        "{exp_name}/{env}/tgt_{target_returns}-guidew_{condition_guidance_w}/{seed}"
    )

    config.env = "OfflineAntRun-v0"
    config.dataset = "dsrl"
    config.returns_condition = False
    config.cost_returns_condition = True
    config.env_ts_condition = True
    
    config.target_returns = "700.0,10.0"
    config.cost_limit = 10.0

    config.max_traj_length = 200
    config.horizon = 1

    config.eval_period = 50
    config.eval_n_trajs = 10
    config.num_eval_envs = 10

    # data aug configs
    config.aug_percent = 0.2
    config.aug_deg = 3
    config.aug_max_rew_decrease = 150.0
    config.aug_beta = 1.0
    config.aug_max_reward = 1000.0
    config.aug_min_reward = 1.0

    config.n_epochs = 5000
    config.n_train_step_per_epoch = 1000

    config.save_period = 0

    # special variable for cdbc
    config.architecture: str = "mlp"

    return config


