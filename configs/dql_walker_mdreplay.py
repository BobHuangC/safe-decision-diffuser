# ml_collections ConfigDict file
import importlib

from ml_collections import ConfigDict

from utilities.utils import WandBLogger


def get_config():
    config = ConfigDict()

    config.trainer = "DiffusionQLTrainer"
    config.type = "model-free"

    config.env = "walker2d-medium-replay-v2"
    config.dataset = "d4rl"
    config.dataset_class = "QLearningDataset"
    config.normalizer = "NoopNormalizer"
    config.max_traj_length = 1000
    config.horizon = 1
    config.returns_scale = 1.0
    config.include_cost_returns = False
    config.termination_penalty = 0.0

    config.seed = 42
    config.algo_cfg = getattr(
        importlib.import_module("diffuser.trainer"), config.trainer
    ).get_default_config()
    config.batch_size = 256
    config.reward_scale = 1
    config.reward_bias = 0
    config.clip_action = 0.999
    config.encoder_arch = "64-64"
    config.policy_arch = "256-256-256"
    config.qf_arch = "256-256-256"
    config.orthogonal_init = False
    config.policy_log_std_multiplier = 1.0
    config.policy_log_std_offset = -1.0

    config.n_epochs = 2000
    config.n_train_step_per_epoch = 1000
    config.eval_period = 10
    config.eval_n_trajs = 10
    config.num_eval_envs = 10
    config.eval_env_seed = 0

    config.qf_layer_norm = False
    config.policy_layer_norm = False
    config.activation = "mish"
    config.obs_norm = False
    config.act_method = ""
    config.sample_method = "ddpm"
    config.policy_temp = 1.0
    config.norm_reward = False

    config.save_model = False
    config.logging = WandBLogger.get_default_config()
    return config
