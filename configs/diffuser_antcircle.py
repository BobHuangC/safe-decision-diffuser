# ml_collections ConfigDict file
import importlib
from ml_collections import ConfigDict

from utilities.utils import WandBLogger


def get_config():
    config = ConfigDict()

    config.trainer = "DiffuserTrainer"
    config.type = "model-free"

    config.env = "OfflineAntCircle-v0"
    config.dataset = "dsrl"
    config.dataset_class = "SequenceDataset"
    config.max_traj_length = 1000
    config.horizon = 100
    config.include_cost_returns = False
    config.returns_scale = 400.0
    config.termination_penalty = -100.0

    config.seed = 42
    config.algo_cfg = getattr(
        importlib.import_module("diffuser.trainer"), config.trainer
    ).get_default_config()
    config.algo_cfg.horizon = config.horizon
    config.batch_size = 256
    config.reward_scale = 1
    config.reward_bias = 0
    config.cost_scale = 1
    config.cost_bias = 0
    config.clip_action = 0.999
    config.dim = 128
    # config.dim_mults = "1-4-8"
    config.dim_mults = "1-2-4"
    config.inv_hidden_dims = "256-256"
    config.kernel_size = 5
    config.returns_condition = True
    config.condition_guidence_w = 1.2
    config.condition_dropout = 0.1

    config.n_epochs = 2000
    config.n_train_step_per_epoch = 1000
    config.eval_period = 10
    config.eval_n_trajs = 10

    config.activation = "mish"
    config.act_method = "ddpm"
    config.sample_method = "ddpm"
    config.policy_temp = 1.0
    config.norm_reward = False
    config.cost_reward = False
    config.norm_cost = False

    config.save_model = False
    config.logging = WandBLogger.get_default_config()
    return config
