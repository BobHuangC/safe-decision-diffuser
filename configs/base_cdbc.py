from ml_collections import ConfigDict, config_dict

from utilities.utils import WandBLogger


def get_base_config():
    config = ConfigDict()
    config.exp_name = config_dict.required_placeholder(str)
    config.log_dir_format = config_dict.required_placeholder(str)

    config.trainer = "CondDiffusionBCTrainer"
    config.type = "model-free"

    config.env = config_dict.required_placeholder(str)
    config.dataset = config_dict.required_placeholder(str)
    config.dataset_class = "QLearningDataset"
    config.use_padding = True
    config.normalizer = "LimitsNormalizer"
    config.max_traj_length = config_dict.required_placeholder(int)
    config.horizon = 1
    config.returns_condition = True
    config.cost_returns_condition = True
    config.env_ts_condition = True
    config.termination_penalty = 0.0
    config.target_returns = config_dict.required_placeholder(str)

    # data aug configs
    config.aug_percent = 0.3
    config.aug_deg = 2
    config.aug_max_rew_decrease = 100.0
    config.aug_beta = 1.0
    config.aug_max_reward = 500.0
    config.aug_min_reward = 1.0

    config.aug_pareto_optimal_only = False
    config.aug_rmin = 0
    config.aug_cost_bins = 60
    config.aug_max_num_per_bin = 1

    config.seed = 300
    config.batch_size = 256
    config.discount = 1.0
    config.policy_arch = "256-256-256"
    config.condition_guidance_w = 1.5
    config.condition_dropout = 0.25

    config.n_epochs = config_dict.required_placeholder(int)
    config.n_train_step_per_epoch = config_dict.required_placeholder(int)

    config.evaluator_class = "OnlineEvaluator"
    config.eval_period = 100
    config.eval_n_trajs = 10
    config.num_eval_envs = 10
    config.eval_env_seed = 0

    config.policy_layer_norm = False
    config.activation = "mish"
    config.act_method = "ddpm"
    config.sample_method = "ddpm"

    config.save_period = 0
    config.logging = WandBLogger.get_default_config()

    config.algo_cfg = ConfigDict()
    config.algo_cfg.discount = 0.99
    config.algo_cfg.tau = 0.005
    config.algo_cfg.policy_tgt_freq = 5
    config.algo_cfg.sample_temperature = 0.5
    config.algo_cfg.num_timesteps = 100
    config.algo_cfg.schedule_name = "linear"
    config.algo_cfg.time_embed_size = 16

    # learning related
    config.algo_cfg.lr = 3e-4
    config.algo_cfg.lr_decay = False
    config.algo_cfg.lr_decay_steps = 1000000
    config.algo_cfg.max_grad_norm = 0.0
    config.algo_cfg.weight_decay = 0.0

    # for dpm-solver
    config.algo_cfg.dpm_steps = 15
    config.algo_cfg.dpm_t_end = 0.001

    config.mode = "train" # or "eval"

    return config
