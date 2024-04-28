import torch

from diffuser.algos import CondDiffusionBC
from diffuser.diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from diffuser.hps import hyperparameters
from diffuser.nets import DiffusionPolicy
from diffuser.policy import SamplerPolicy
from diffuser.trainer.base_trainer import BaseTrainer
from utilities.data_utils import cycle, numpy_collate
from utilities.utils import set_random_seed, str_to_list, to_arch


class CondDiffusionBCTrainer(BaseTrainer):
    def _setup(self):
        set_random_seed(self._cfgs.seed)
        # setup logger
        self._wandb_logger = self._setup_logger()

        # setup dataset and eval_sample
        self.dataset, self.eval_sampler = self._setup_dataset()
        target_returns = str_to_list(self._cfgs.target_returns)
        # assert len(target_returns) == 2, target_returns
        assert len(target_returns) % 2 == 0, target_returns
        self.eval_sampler.set_target_returns(target_returns)
        if hasattr(self.eval_sampler.env, "set_target_cost"):
            self.eval_sampler.env.set_target_cost(target_returns[1])
        data_sampler = torch.utils.data.RandomSampler(self.dataset)
        self._dataloader = cycle(
            torch.utils.data.DataLoader(
                self.dataset,
                sampler=data_sampler,
                batch_size=self._cfgs.batch_size,
                collate_fn=numpy_collate,
                drop_last=True,
                num_workers=8,
            )
        )

        # setup policy
        self._policy = self._setup_policy()

        # setup agent
        self._cfgs.algo_cfg.max_grad_norm = hyperparameters[self._cfgs.env]["gn"]
        self._agent = CondDiffusionBC(self._cfgs.algo_cfg, self._policy)

        # setup sampler policy
        sampler_policy = SamplerPolicy(self._agent.policy)
        self._evaluator = self._setup_evaluator(
            sampler_policy, self.eval_sampler, self.dataset
        )

    def _reset_target_returns(self, new_target_returns):
        target_returns = str_to_list(new_target_returns)
        assert len(target_returns) % 2 == 0, target_returns
        self.eval_sampler.set_target_returns(target_returns)
        if hasattr(self.eval_sampler.env, "set_target_cost"):
            self.eval_sampler.env.set_target_cost(target_returns[1])

        # setup sampler policy
        sampler_policy = SamplerPolicy(self._agent.policy)
        self._evaluator = self._setup_evaluator(
            sampler_policy, self.eval_sampler, self.dataset
        )

    def _setup_policy(self):
        gd = GaussianDiffusion(
            num_timesteps=self._cfgs.algo_cfg.num_timesteps,
            schedule_name=self._cfgs.algo_cfg.schedule_name,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            env_ts_condition=self._cfgs.env_ts_condition,
            returns_condition=self._cfgs.returns_condition,
            cost_returns_condition=self._cfgs.cost_returns_condition,
            # min_value=-self._max_action,
            # max_value=self._max_action,
            sample_temperature=self._cfgs.algo_cfg.sample_temperature,
        )
        if self._cfgs.architecture == "mlp":
            policy = DiffusionPolicy(
                diffusion=gd,
                observation_dim=self._observation_dim,
                action_dim=self._action_dim,
                arch=to_arch(self._cfgs.policy_arch),
                time_embed_size=self._cfgs.algo_cfg.time_embed_size,
                use_layer_norm=self._cfgs.policy_layer_norm,
                sample_method=self._cfgs.sample_method,
                dpm_steps=self._cfgs.algo_cfg.dpm_steps,
                dpm_t_end=self._cfgs.algo_cfg.dpm_t_end,
                env_ts_condition=self._cfgs.env_ts_condition,
                returns_condition=self._cfgs.returns_condition,
                cost_returns_condition=self._cfgs.cost_returns_condition,
                condition_dropout=self._cfgs.condition_dropout,
            )
        elif self._cfgs.architecture == "transformer":
            raise NotImplementedError

        return policy
