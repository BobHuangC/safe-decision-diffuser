from ml_collections import ConfigDict

from utilities.utils import WandBLogger


def get_config():
    config = ConfigDict()
    config.exp_name = "diffuser_inv_dsrl"
    config.log_dir_format = "{exp_name}/{env}/h_{horizon}-tgt_{target_returns}-usecost_{cost_returns_condition}-guidew_{condition_guidance_w}/{seed}"

    config.trainer = "DiffuserTrainer"
    config.type = "model-free"

    config.env = "OfflineCarRun-v0"
    config.dataset = "dsrl"
    config.dataset_class = "SequenceDataset"
    config.use_padding = True
    config.normalizer = "LimitsNormalizer"
    config.max_traj_length = 200
    config.horizon = 12
    config.returns_condition = True
    config.cost_returns_condition = True
    config.termination_penalty = 0.0
    config.target_returns = "575.0,10.0"

    # data aug configs
    config.aug_percent = 0.3
    config.aug_deg = 0
    config.aug_max_rew_decrease = 100.0
    config.aug_beta = 1.0
    config.aug_max_reward = 600.0
    config.aug_min_reward = 1.0

    config.aug_pareto_optimal_only = False
    config.aug_rmin = 0
    config.aug_cost_bins = 60
    config.aug_max_num_per_bin = 1

    config.seed = 100
    config.batch_size = 256
    config.clip_action = 0.999
    config.dim = 64
    config.dim_mults = "1-2-4"
    config.kernel_size = 5
    config.condition_guidance_w = 1.2
    config.condition_dropout = 0.25

    config.use_inv_dynamic = True
    config.inv_hidden_dims = "256-256"

    config.n_epochs = 500
    config.n_train_step_per_epoch = 1000

    config.evaluator_class = "OnlineEvaluator"
    config.eval_period = 10
    config.eval_n_trajs = 20
    config.num_eval_envs = 20
    config.eval_env_seed = 0

    config.activation = "mish"
    config.act_method = "ddpm"
    config.sample_method = "ddpm"

    config.save_period = 0
    config.logging = WandBLogger.get_default_config()

    config.algo_cfg = ConfigDict()
    config.algo_cfg.horizon = config.horizon
    config.algo_cfg.loss_discount = 1.0
    config.algo_cfg.sample_temperature = 0.5
    config.algo_cfg.num_timesteps = 100
    config.algo_cfg.schedule_name = "cosine"
    # learning related
    config.algo_cfg.lr = 2e-4
    config.algo_cfg.lr_decay = False
    config.algo_cfg.lr_decay_steps = 1000000
    config.algo_cfg.max_grad_norm = 0.0
    config.algo_cfg.weight_decay = 0.0
    # for dpm-solver
    config.algo_cfg.dpm_steps = 15
    config.algo_cfg.dpm_t_end = 0.001
    # for ema decay
    config.algo_cfg.ema_decay = 0.995
    config.algo_cfg.step_start_ema = 2000
    config.algo_cfg.update_ema_every = 10

    return config