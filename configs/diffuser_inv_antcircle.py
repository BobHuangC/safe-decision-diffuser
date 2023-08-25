from ml_collections import ConfigDict

from utilities.utils import WandBLogger


def get_config():
    config = ConfigDict()
    config.exp_name = "diffuser_inv_dsrl"
    config.log_dir_format = "{exp_name}/{env}/h_{horizon}-r_{returns_scale}-guidew_{condition_guidance_w}/{seed}"

    config.trainer = "DiffuserTrainer"
    config.type = "model-free"

    config.env = "OfflineAntCircle-v0"
    config.dataset = "dsrl"
    config.dataset_class = "SequenceDataset"
    config.use_padding = True
    config.normalizer = "LimitsNormalizer"
    config.max_traj_length = 1000
    config.horizon = 100
    config.include_cost_returns = True
    config.returns_scale = 400.0
    config.termination_penalty = -100.0

    config.dataAugmentation_method = "Augmentation"
    config.dataAugment_percent = 0.3
    config.dataAug_deg = 3
    config.dataAug_max_rew_decrease = 1
    config.dataAug_beta = 1
    config.dataAug_max_reward = 1000.0
    config.dataAug_min_reward = 0.0
    config.dataAug_aug_rmin = 0
    config.dataAug_aug_rmax = 600
    config.dataAug_aug_cmin = 5
    config.dataAug_aug_cmax = 50
    config.dataAug_cgap = 5
    config.dataAug_rstd = 1
    config.dataAug_cstd = 0.25
    config.dataAug_rmin = 0
    config.dataAug_cost_bins = 60
    config.dataAug_max_num_per_bin = 1

    config.seed = 42
    config.batch_size = 256
    config.reward_scale = 1
    config.reward_bias = 0
    config.cost_scale = 1
    config.cost_bias = 0
    config.clip_action = 0.999
    config.dim = 128
    # config.dim_mults = "1-4-8"
    config.dim_mults = "1-2-4"
    config.kernel_size = 5
    config.returns_condition = True
    config.cost_returns_condition = True
    config.condition_guidance_w = 1.2
    config.condition_dropout = 0.1

    config.use_inv_dynamic = True
    config.inv_hidden_dims = "256-256"

    config.n_epochs = 2000
    config.n_train_step_per_epoch = 1000
    config.eval_period = 10
    config.eval_n_trajs = 10
    config.num_eval_envs = 10
    config.eval_env_seed = 0

    config.activation = "mish"
    config.act_method = "ddpm"
    config.sample_method = "ddpm"
    config.policy_temp = 1.0
    config.cost_reward = False

    config.save_model = False
    config.logging = WandBLogger.get_default_config()

    config.algo_cfg = ConfigDict()
    config.algo_cfg.horizon = config.horizon
    config.algo_cfg.lr = 2e-4
    config.algo_cfg.num_timesteps = 200
    config.algo_cfg.schedule_name = "linear"
    # learning related
    config.algo_cfg.lr = 2e-4
    config.algo_cfg.lr_decay = False
    config.algo_cfg.lr_decay_steps = 1000000
    config.algo_cfg.max_grad_norm = 0.0
    config.algo_cfg.weight_decay = 0.0
    # for dpm-solver
    config.algo_cfg.dpm_steps = 15
    config.algo_cfg.dpm_t_end = 0.001

    return config
