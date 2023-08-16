import torch

from diffuser.algos import DecisionDiffuser
from diffuser.diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from diffuser.nets import DiffusionPlanner, InverseDynamic
from diffuser.policy import DiffuserPolicy
from diffuser.trainer.base_trainer import BaseTrainer
from utilities.data_utils import cycle, numpy_collate
from utilities.utils import set_random_seed, to_arch


class DiffuserTrainer(BaseTrainer):
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
        self._agent = DecisionDiffuser(
            self._cfgs.algo_cfg, self._planner, self._inv_model
        )

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
            cost_returns_condition=self._cfgs.cost_returns_condition,
            condition_guidance_w=self._cfgs.condition_guidance_w,
        )
        planner = DiffusionPlanner(
            diffusion=gd,
            horizon=self._cfgs.horizon,
            observation_dim=self._observation_dim,
            dim=self._cfgs.dim,
            dim_mults=to_arch(self._cfgs.dim_mults),
            returns_condition=self._cfgs.returns_condition,
            cost_returns_condition=self._cfgs.cost_returns_condition,
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
            self._sampler_policy.update_params(self._agent.train_params),
            self._cfgs.eval_n_trajs,
            deterministic=True,
        )
        return trajs
