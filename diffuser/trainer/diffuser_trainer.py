from functools import partial
import torch
import jax
import jax.numpy as jnp

from diffuser.algos import DecisionDiffuser
from diffuser.diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from diffuser.nets import DiffusionPlanner, InverseDynamic
from diffuser.policy import DiffuserPolicy
from diffuser.trainer.base_trainer import BaseTrainer
from utilities.data_utils import cycle, numpy_collate
from utilities.utils import set_random_seed, to_arch
from utilities.jax_utils import batch_to_jax, next_rng


class DiffuserTrainer(BaseTrainer):
    def _setup(self):
        set_random_seed(self._cfgs.seed)
        # setup logger
        self._wandb_logger = self._setup_logger()

        # setup dataset and eval_sample
        dataset, self._eval_sampler = self._setup_dataset()
        sampler = torch.utils.data.RandomSampler(dataset)
        self._dataloader = cycle(
            torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self._cfgs.batch_size,
                collate_fn=numpy_collate,
                drop_last=True,
                num_workers=8,
            )
        )

        if self._cfgs.eval_mode == "offline":
            eval_sampler = torch.utils.data.RandomSampler(dataset)
            self._eval_dataloader = cycle(
                torch.utils.data.DataLoader(
                    dataset,
                    sampler=eval_sampler,
                    batch_size=self._cfgs.eval_batch_size,
                    collate_fn=numpy_collate,
                    drop_last=True,
                    num_workers=4,
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
            sample_temperature=self._cfgs.algo_cfg.sample_temperature,
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

    def _offline_evaluate(self):
        eval_batch = batch_to_jax(next(self._eval_dataloader))
        rng = next_rng()
        return self._offline_eval_step(
            self._agent.train_states, rng, eval_batch
        )

    @partial(jax.jit, static_argnames=("self"))
    def _offline_eval_step(self, train_states, rng, eval_batch):
        metrics = {}

        samples = eval_batch["samples"]
        conditions = eval_batch["conditions"]
        returns = eval_batch["returns"]
        actions = eval_batch["actions"][:, 0]

        pred_actions = self._inv_model.apply(
            train_states["inv_model"].params,
            jnp.concatenate([samples[:, 0], samples[:, 1]], axis=-1),
        )
        pred_act_mse = jnp.mean(jnp.square(pred_actions - actions))

        plan_observations = self._planner.apply(
            train_states["planner"].params,
            rng,
            conditions=conditions,
            returns=returns,
            method=self._planner.ddpm_sample,
        )
        obs_comb = jnp.concatenate(
            [plan_observations[:, 0], plan_observations[:, 1]], axis=-1
        )
        plan_actions = self._inv_model.apply(
            train_states["inv_model"].params,
            obs_comb,
        )

        plan_obs_mse = jnp.mean(
            jnp.square(plan_observations - samples)
        )
        plan_act_mse = jnp.mean(jnp.square(plan_actions - actions))

        metrics["plan_obs_mse"] = plan_obs_mse
        metrics["plan_act_mse"] = plan_act_mse
        metrics["pred_act_mse"] = pred_act_mse
        return metrics
