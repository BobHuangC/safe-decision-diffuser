from configs.base_tcdbc import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "tcdbc_dsrl"
    config.log_dir_format = "{exp_name}/{env}/{architecture}-gw_{condition_guidance_w}-cdp_{condition_dropout}/{seed}"
    config.eval_log_dir_format = "{log_dir_format}/eval"

    config.env = "OfflineAntRun-v0"
    config.dataset = "dsrl"
    config.returns_condition = True
    config.cost_returns_condition = True
    config.env_ts_condition = True
    config.condition_guidance_w = 1.2
    config.condition_dropout = 0.2

    config.target_returns = "700.0,10,750,20,800,40"
    config.cost_limit = 10.0

    config.max_traj_length = 200
    config.horizon = 1

    config.eval_period = 25
    config.eval_n_trajs = 10
    config.num_eval_envs = 10

    # data aug configs
    config.aug_percent = 0.2
    config.aug_deg = 3
    config.aug_max_rew_decrease = 150.0
    config.aug_beta = 1.0
    config.aug_max_reward = 1000.0
    config.aug_min_reward = 1.0

    config.batch_size = 2048

    config.n_epochs = 2000
    config.n_train_step_per_epoch = 1000

    config.save_period = 25

    config.architecture = "transformer"
    config.algo_cfg.transformer_n_heads = 4
    config.algo_cfg.transformer_depth = 1
    config.algo_cfg.transformer_dropout = 0.0
    config.algo_cfg.transformer_embedding_dim = 128

    # evaluate_pro config
    config.eval_target_reward_returns_list = (
        "500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0"
    )
    config.eval_target_cost_returns_list = (
        "0.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0"
    )

    # mode represents whether the config is used for training or evaluation
    config.mode = "train"  # or "eval"

    # learning related
    config.algo_cfg.lr = 1e-4
    config.algo_cfg.lr_decay = False
    config.algo_cfg.lr_decay_steps = 300
    config.algo_cfg.lr_decay_alpha = 0.05

    return config
