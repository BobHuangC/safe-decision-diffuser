from configs.base_cdbc import get_base_config

def get_config():
    config = get_base_config()
    config.exp_name = "cdbc_dsrl"
    config.log_dir_format = (
        "{exp_name}/{env}/tgt_{target_returns}-guidew_{condition_guidance_w}/{seed}"
    )

    config.env = "OfflineAntRun-v0"
    config.dataset = "dsrl"
    config.returns_condition = True
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

    return config





# class CDTTrainConfig:
#     # dataset params
#     outliers_percent: float = None
#     noise_scale: float = None
#     inpaint_ranges: Tuple[Tuple[float, float], ...] = None
#     epsilon: float = None
#     density: float = 1.0
#     # model params
#     embedding_dim: int = 128
#     num_layers: int = 3
#     num_heads: int = 8
#     action_head_layers: int = 1
#     seq_len: int = 10
#     episode_len: int = 300
#     attention_dropout: float = 0.1
#     residual_dropout: float = 0.1
#     embedding_dropout: float = 0.1
#     time_emb: bool = True
#     # training params
#     # task: str = "OfflineCarCircle-v0"
#     # dataset: str = None
#     learning_rate: float = 1e-4
#     betas: Tuple[float, float] = (0.9, 0.999)
#     weight_decay: float = 1e-4
#     clip_grad: Optional[float] = 0.25
#     batch_size: int = 2048
#     update_steps: int = 100_000
#     lr_warmup_steps: int = 500
#     reward_scale: float = 0.1
#     cost_scale: float = 1
#     num_workers: int = 8
#     # evaluation params
#     target_returns: Tuple[Tuple[float, ...],
#                           ...] = ((450.0, 10), (500.0, 20), (550.0, 50))  # reward, cost
#     cost_limit: int = 10
#     eval_episodes: int = 10
#     eval_every: int = 2500
#     # general params
#     seed: int = 0
#     device: str = "cuda:2"
#     threads: int = 6
#     # augmentation param
#     deg: int = 4
#     pf_sample: bool = False
#     beta: float = 1.0
#     augment_percent: float = 0.2
#     # maximum absolute value of reward for the augmented trajs
#     max_reward: float = 600.0
#     # minimum reward above the PF curve
#     min_reward: float = 1.0
#     # the max drecrease of ret between the associated traj
#     # w.r.t the nearest pf traj
#     max_rew_decrease: float = 100.0
#     # model mode params
#     use_rew: bool = True
#     use_cost: bool = True
#     cost_transform: bool = True
#     cost_prefix: bool = False
#     add_cost_feat: bool = False
#     mul_cost_feat: bool = False
#     cat_cost_feat: bool = False
#     loss_cost_weight: float = 0.02
#     loss_state_weight: float = 0
#     cost_reverse: bool = False
#     # pf only mode param
#     pf_only: bool = False
#     rmin: float = 300
#     cost_bins: int = 60
#     npb: int = 5
#     cost_sample: bool = True
#     linear: bool = True  # linear or inverse
#     start_sampling: bool = False
#     prob: float = 0.2
#     stochastic: bool = True
#     init_temperature: float = 0.1
#     no_entropy: bool = False
#     # random augmentation
#     random_aug: float = 0
#     aug_rmin: float = 400
#     aug_rmax: float = 500
#     aug_cmin: float = -2
#     aug_cmax: float = 25
#     cgap: float = 5
#     rstd: float = 1
#     cstd: float = 0.2
