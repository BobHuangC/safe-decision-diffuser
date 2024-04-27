from configs.base_tcdbc import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "tcdbc_dsrl"
    config.log_dir_format = "{exp_name}/{env}/{architecture}-gw_{condition_guidance_w}-cdp_{condition_dropout}-{normalizer}-normret_{normalize_returns}/{seed}/2024-3-11-5"
    # config.log_dir_format = "{exp_name}/{env}/{architecture}-gw_{condition_guidance_w}-cdp_{condition_dropout}-{normalizer}-normret_{normalize_returns}/{seed}/test"
    # config.log_dir_format = "{exp_name}/{env}/{architecture}-gw_{condition_guidance_w}-cdp_{condition_dropout}-{normalizer}-normret_{normalize_returns}/{seed}"
    config.eval_log_dir_format = "{log_dir_format}/eval"

    config.env = "OfflineAntRun-v0"
    config.dataset = "dsrl"
    config.normalizer = "CDFNormalizer"
    config.normalize_returns = True
    config.returns_condition = True
    config.cost_returns_condition = True
    config.env_ts_condition = True
    config.condition_guidance_w = 1.2
    config.condition_dropout = 0.2

    config.target_returns = "700.0,10,750,20,800,40"
    config.cost_limit = 10.0

    config.max_traj_length = 200
    config.horizon = 1

    config.eval_period = 10
    config.eval_n_trajs = 20
    config.num_eval_envs = 10

    # data aug configs
    config.aug_deg = 3
    config.aug_max_rew_decrease = 150.0
    config.aug_max_reward = 1000.0

    config.seed = 300
    config.batch_size = 2048

    config.n_epochs = 2000
    config.n_train_step_per_epoch = 1000

    config.save_period = -1

    config.architecture = "transformer"
    config.algo_cfg.transformer_n_heads = 4
    config.algo_cfg.transformer_depth = 1
    config.algo_cfg.transformer_dropout = 0.0
    config.algo_cfg.transformer_embedding_dim = 32

    # learning related
    config.algo_cfg.lr = 1e-4
    config.algo_cfg.lr_decay = True
    config.algo_cfg.lr_decay_steps = 300
    config.algo_cfg.lr_decay_alpha = 0.01

    return config
