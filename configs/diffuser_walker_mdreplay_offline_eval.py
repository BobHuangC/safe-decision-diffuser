from ml_collections import ConfigDict

from utilities.utils import WandBLogger


def get_config():
    config = ConfigDict()
    config.exp_name = "diffuser_d4rl_offline"
    config.log_dir_format = "{exp_name}/{env}/h_{horizon}-r_{returns_scale}-guidew_{condition_guidance_w}/{seed}"

    config.trainer = "DiffuserTrainer"
    config.type = "model-free"

    config.env = "walker2d-medium-replay-v2"
    config.dataset = "d4rl"
    config.dataset_class = "SequenceDataset"
    config.use_padding = True
    config.normalizer = "LimitsNormalizer"
    config.max_traj_length = 1000
    config.horizon = 100
    config.include_returns = True
    config.include_cost_returns = False
    config.returns_scale = 400.0
    config.termination_penalty = -100.0

    config.seed = 100
    config.batch_size = 32
    config.reward_scale = 1
    config.reward_bias = 0
    config.clip_action = 0.999
    config.dim = 128
    config.dim_mults = "1-4-8"
    config.inv_hidden_dims = "256-256"
    config.kernel_size = 5
    config.returns_condition = True
    config.cost_returns_condition = False
    config.condition_guidance_w = 1.2
    config.condition_dropout = 0.25

    config.n_epochs = 1000
    config.n_train_step_per_epoch = 1000

    config.evaluator_class = "DiffuserOfflineEvaluator"
    config.eval_batch_size = 128
    config.eval_period = 1
    config.num_eval_envs = 1
    config.eval_env_seed = 0

    config.activation = "mish"
    config.act_method = "ddpm"
    config.sample_method = "ddpm"
    config.norm_reward = False

    config.save_period = 100
    config.logging = WandBLogger.get_default_config()

    config.algo_cfg = ConfigDict()
    config.algo_cfg.horizon = config.horizon
    config.algo_cfg.loss_discount = 1.0
    config.algo_cfg.lr = 2e-4
    config.algo_cfg.num_timesteps = 200
    config.algo_cfg.schedule_name = "cosine"
    config.algo_cfg.sample_temperature = 0.5
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
