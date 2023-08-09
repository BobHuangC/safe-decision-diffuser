import importlib
import torch
import gymnasium
import numpy as np
from ml_collections import ConfigDict

from data.dataset import get_dsrl_dataset
from utilities.utils import set_random_seed, to_arch
from utilities.sampler import TrajSampler
from utilities.data_utils import cycle, numpy_collate
from diffuser.trainer.base_trainer import BaseTrainer
from diffuser.policy import DiffuserPolicy
from diffuser.nets import DiffusionPlanner, InverseDynamic
from diffuser.algos import DecisionDiffuser
from diffuser.diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)


class DiffuserTrainer(BaseTrainer):
    @staticmethod
    def get_default_config(updates=None):
        cfg = ConfigDict()
        cfg.discount = 0.99
        cfg.tau = 0.005
        cfg.policy_tgt_freq = 5
        cfg.num_timesteps = 100
        cfg.schedule_name = "linear"
        cfg.alpha = 2.0  # NOTE 0.25 in diffusion rl but 2.5 in td3
        cfg.use_pred_xstart = True

        # learning related
        cfg.lr = 3e-4
        cfg.diff_coef = 1.0
        cfg.guide_coef = 1.0
        cfg.lr_decay = False
        cfg.lr_decay_steps = 1000000
        cfg.max_grad_norm = 0.0
        cfg.weight_decay = 0.0

        # for dpm-solver
        cfg.dpm_steps = 15
        cfg.dpm_t_end = 0.001

        if updates is not None:
            cfg.update(ConfigDict(updates).copy_and_resolve_references())
        return cfg

    def _setup(self):
        set_random_seed(self._cfgs.seed)
        # setup logger
        self._wandb_logger = self._setup_logger()

        # setup dataset and eval_sample
        dataset, self._eval_sampler = self._setup_dataset()
        self._dataloader = cycle(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self._cfgs.batch_size,
                collate_fn=numpy_collate,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )

        # setup policy
        self._planner, self._inv_model = self._setup_policy()

        # setup agent
        self._agent = DecisionDiffuser(self._cfgs.algo_cfg, self._planner, self._inv_model)

        # setup sampler policy
        self._sampler_policy = DiffuserPolicy(self._planner, self._inv_model)

    def _setup_policy(self):
        gd = GaussianDiffusion(
            num_timesteps=self._cfgs.algo_cfg.num_timesteps,
            schedule_name=self._cfgs.algo_cfg.schedule_name,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            returns_condition=self._cfgs.returns_condition,
            condition_guidence_w=self._cfgs.condition_guidence_w,
        )
        planner = DiffusionPlanner(
            diffusion=gd,
            horizon=self._cfgs.horizon,
            observation_dim=self._observation_dim,
            dim=self._cfgs.dim,
            dim_mults=to_arch(self._cfgs.dim_mults),
            returns_condition=self._cfgs.returns_condition,
            condition_dropout=self._cfgs.condition_dropout,
            kernel_size=self._cfgs.kernel_size,
            sample_method=self._cfgs.sample_method,
            dpm_steps=self._cfgs.algo_cfg.dpm_steps,
            dpm_t_end=self._cfgs.algo_cfg.dpm_t_end,
        )
        inv_model = InverseDynamic(
            action_dim=self._action_dim,
            hidden_dims=to_arch(self._cfgs.inv_hidden_dims),
        )
        return planner, inv_model

    def _sample_trajs(self, act_method: str):
        self._sampler_policy.act_method = act_method
        trajs = self._eval_sampler.sample(
            self._sampler_policy.update_params(
                self._agent.train_params
            ),
            self._cfgs.eval_n_trajs,
            deterministic=True,
        )
        return trajs

    def _setup_dsrl(self):
        eval_sampler = TrajSampler(gymnasium.make(self._cfgs.env), self._cfgs.max_traj_length)

        norm_reward = self._cfgs.norm_reward
        if "antmaze" in self._cfgs.env:
            norm_reward = False

        if self._cfgs.dataset_class in ["QLearningDataset"]:
            include_next_obs = True
        else:
            include_next_obs = False

        dataset = get_dsrl_dataset(
            eval_sampler.env,
            max_traj_length=self._cfgs.max_traj_length,
            norm_reward=norm_reward,
            include_next_obs=include_next_obs,
            termination_penalty=self._cfgs.termination_penalty,
        )
        dataset["rewards"] = (
            dataset["rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
        )
        dataset["actions"] = np.clip(
            dataset["actions"], -self._cfgs.clip_action, self._cfgs.clip_action
        )

        dataset = getattr(importlib.import_module("data.sequence"), self._cfgs.dataset_class)(
            dataset,
            returns_scale=self._cfgs.returns_scale,
            horizon=self._cfgs.horizon,
            max_traj_length=self._cfgs.max_traj_length,
        )
        eval_sampler.set_normalizer(dataset.normalizer)
        return dataset, eval_sampler
