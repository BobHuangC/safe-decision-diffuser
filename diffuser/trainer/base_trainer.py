# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from collections import deque

import absl
import absl.flags
import gym
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from diffuser.constants import (
    DATASET,
    DATASET_ABBR_MAP,
    DATASET_MAP,
    ENV_MAP,
    ENVNAME_MAP,
)
from diffuser.hps import hyperparameters
from utilities.jax_utils import batch_to_jax
from utilities.sampler import TrajSampler
from utilities.utils import Timer, WandBLogger, get_user_flags, prefix_metrics
from viskit.logging import logger, setup_logger


class BaseTrainer:
    def __init__(self, config):
        self._cfgs = absl.flags.FLAGS

        self._cfgs.algo_cfg.max_grad_norm = hyperparameters[self._cfgs.env]["gn"]
        self._cfgs.algo_cfg.lr_decay_steps = (
            self._cfgs.n_epochs * self._cfgs.n_train_step_per_epoch
        )

        if self._cfgs.activation == "mish":
            act_fn = lambda x: x * jnp.tanh(jax.nn.softplus(x))
        else:
            act_fn = getattr(jax.nn, self._cfgs.activation)

        self._act_fn = act_fn

        self._variant = get_user_flags(self._cfgs, config)
        for k, v in self._cfgs.algo_cfg.items():
            self._variant[f"algo.{k}"] = v

        # get high level env
        env_name_full = self._cfgs.env
        for scenario_name in ENV_MAP:
            if scenario_name in env_name_full:
                self._env = ENV_MAP[scenario_name]
                break
        else:
            raise NotImplementedError

    def train(self):
        self._setup()

        act_methods = self._cfgs.act_method.split("-")
        viskit_metrics = {}
        recent_returns = {method: deque(maxlen=10) for method in act_methods}
        best_returns = {method: -float("inf") for method in act_methods}
        for epoch in range(self._cfgs.n_epochs):
            metrics = {"epoch": epoch}

            with Timer() as train_timer:
                for _ in tqdm.tqdm(range(self._cfgs.n_train_step_per_epoch)):
                    batch = batch_to_jax(next(self._dataloader))
                    metrics.update(prefix_metrics(self._agent.train(batch), "agent"))

            with Timer() as eval_timer:
                if epoch == 0 or (epoch + 1) % self._cfgs.eval_period == 0:
                    for method in act_methods:
                        trajs = self._sample_trajs(method)

                        post = "" if len(act_methods) == 1 else "_" + method
                        metrics["average_return" + post] = np.mean(
                            [np.sum(t["rewards"]) for t in trajs]
                        )
                        metrics["average_traj_length" + post] = np.mean(
                            [len(t["rewards"]) for t in trajs]
                        )
                        metrics[
                            "average_normalizd_return" + post
                        ] = cur_return = np.mean(
                            [
                                self._eval_sampler.env.get_normalized_score(
                                    np.sum(t["rewards"])
                                )
                                for t in trajs
                            ]
                        )
                        recent_returns[method].append(cur_return)
                        metrics["average_10_normalized_return" + post] = np.mean(
                            recent_returns[method]
                        )
                        metrics["best_normalized_return" + post] = best_returns[
                            method
                        ] = max(best_returns[method], cur_return)
                        metrics["done" + post] = np.mean(
                            [np.sum(t["dones"]) for t in trajs]
                        )

                    if self._cfgs.save_model:
                        save_data = {
                            "agent": self._agent,
                            "variant": self._variant,
                            "epoch": epoch,
                        }
                        self._wandb_logger.save_pickle(save_data, f"model_{epoch}.pkl")

            metrics["train_time"] = train_timer()
            metrics["eval_time"] = eval_timer()
            metrics["epoch_time"] = train_timer() + eval_timer()
            self._wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # save model
        if self._cfgs.save_model:
            save_data = {"agent": self._agent, "variant": self._variant, "epoch": epoch}
            self._wandb_logger.save_pickle(save_data, "model_final.pkl")

    def _setup(self):
        raise NotImplementedError

    def _sample_trajs(self):
        raise NotImplementedError

    def _setup_logger(self):
        env_name_high = ENVNAME_MAP[self._env]
        env_name_full = self._cfgs.env
        dataset_name_abbr = DATASET_ABBR_MAP[self._cfgs.dataset]

        logging_configs = self._cfgs.logging
        logging_configs[
            "project"
        ] = f"{self._cfgs.trainer}-{env_name_high}-{dataset_name_abbr}"
        wandb_logger = WandBLogger(
            config=logging_configs, variant=self._variant, env_name=env_name_full
        )
        setup_logger(
            variant=self._variant,
            base_log_dir=self._cfgs.logging.output_dir,
            exp_id=wandb_logger.experiment_id,
            seed=self._cfgs.seed,
            include_exp_prefix_sub_dir=False,
        )
        return wandb_logger

    def _setup_d4rl(self):
        from data.d4rl import get_dataset

        if self._cfgs.dataset_class in ["QLearningDataset"]:
            include_next_obs = True
        else:
            include_next_obs = False

        eval_sampler = TrajSampler(gym.make(self._cfgs.env), self._cfgs.max_traj_length)
        dataset = get_dataset(
            eval_sampler.env,
            max_traj_length=self._cfgs.max_traj_length,
            norm_reward=self._cfgs.norm_reward,
            include_next_obs=include_next_obs,
            termination_penalty=self._cfgs.termination_penalty,
        )
        return dataset, eval_sampler

    def _setup_dsrl(self):
        from data.dsrl import get_dataset

        if self._cfgs.dataset_class in ["QLearningDataset"]:
            include_next_obs = True
        else:
            include_next_obs = False

        eval_sampler = TrajSampler(
            gymnasium.make(self._cfgs.env), self._cfgs.max_traj_length
        )
        dataset = get_dataset(
            eval_sampler.env,
            max_traj_length=self._cfgs.max_traj_length,
            use_cost=self._cfgs.include_cost_returns,
            norm_reward=self._cfgs.norm_reward,
            norm_cost=self._cfgs.norm_cost,
            termination_penalty=self._cfgs.termination_penalty,
            include_next_obs=include_next_obs,
        )
        return dataset, eval_sampler

    def _setup_dataset(self):
        dataset_type = DATASET_MAP[self._cfgs.dataset]
        if dataset_type == DATASET.D4RL:
            dataset, eval_sampler = self._setup_d4rl()
        elif dataset_type == DATASET.DSRL:
            dataset, eval_sampler = self._setup_dsrl()
        else:
            raise NotImplementedError

        dataset["rewards"] = (
            dataset["rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
        )
        dataset["actions"] = np.clip(
            dataset["actions"], -self._cfgs.clip_action, self._cfgs.clip_action
        )

        dataset = getattr(
            importlib.import_module("data.sequence"), self._cfgs.dataset_class
        )(
            dataset,
            horizon=self._cfgs.horizon,
            max_traj_length=self._cfgs.max_traj_length,
            include_cost_returns=self._cfgs.include_cost_returns,
        )
        eval_sampler.set_normalizer(dataset.normalizer)

        self._observation_dim = eval_sampler.env.observation_space.shape[0]
        self._action_dim = eval_sampler.env.action_space.shape[0]
        self._max_action = float(eval_sampler.env.action_space.high[0])

        return dataset, eval_sampler
