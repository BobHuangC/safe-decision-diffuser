from configs.base_tcdbc import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "tcdbc_dsrl"
    config.log_dir_format = "{exp_name}/{env}/{architecture}-gw_{condition_guidance_w}-cdp_{condition_dropout}-{normalizer}-normret_{normalize_returns}/{seed}/2024-4-6-1"
    # config.log_dir_format = "{exp_name}/{env}/{architecture}-gw_{condition_guidance_w}-cdp_{condition_dropout}-{normalizer}-normret_{normalize_returns}/{seed}/test"
    config.eval_log_dir_format = "{log_dir_format}/eval"

    config.env = "OfflineCarCircle-v0"
    config.dataset = "dsrl"
    config.normalizer = "CDFNormalizer"
    config.normalize_returns = True
    config.returns_condition = True
    config.cost_returns_condition = True
    config.env_ts_condition = True
    config.condition_guidance_w = 1.2
    config.condition_dropout = 0.25

    config.target_returns = "450.0,10,500.0,20,550.0,50"
    config.cost_limit = 10.0

    config.max_traj_length = 300
    config.horizon = 1

    config.eval_period = 10
    config.eval_n_trajs = 20
    config.num_eval_envs = 10

    # data aug configs

    config.seed = 100
    config.batch_size = 2048

    config.n_epochs = 2000
    config.n_train_step_per_epoch = 1000

    config.save_period = 10

    config.architecture = "transformer"
    config.algo_cfg.transformer_n_heads = 4
    config.algo_cfg.transformer_depth = 1
    config.algo_cfg.transformer_dropout = 0
    config.algo_cfg.transformer_embedding_dim = 32

    # learning related
    config.algo_cfg.lr = 1e-4
    config.algo_cfg.lr_decay = False
    config.algo_cfg.lr_decay_steps = 200
    config.algo_cfg.lr_decay_alpha = 0.01

    config.algo_cfg.weight_decay = 1e-4

    # for ema decay
    config.algo_cfg.ema_decay = 0.999
    config.algo_cfg.step_start_ema = 400
    config.algo_cfg.update_ema_every = 10

    return config
