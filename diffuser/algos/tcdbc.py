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

from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import optax

from diffuser.diffusion import GaussianDiffusion
from utilities.flax_utils import TrainState
from utilities.jax_utils import next_rng, value_and_multi_grad

from .base_algo import Algo


def update_target_network(main_params, target_params, tau):
    return jax.tree_map(
        lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
    )


# compatible for both MLP and Transformer
class TransformerCondDiffusionBC(Algo):
    def __init__(self, cfg, policy):
        self.config = cfg
        self.policy = policy
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim
        self.diffusion: GaussianDiffusion = self.policy.diffusion

        self._total_steps = 0
        self._train_states = {}
        # architecture transformer1 corresponds to the TransformerCondPolicyNet1
        if self.policy.architecture == "transformer1":
            policy_params = self.policy.init(
                next_rng(),
                next_rng(),
                observations=jnp.zeros((10, self.observation_dim)),
                actions=jnp.zeros((10, self.action_dim)),
                observation_conditions={},
                ts=jnp.zeros((10,), dtype=jnp.int32),  # ts
                env_ts=jnp.zeros((10,), dtype=jnp.int32),
                returns_to_go=jnp.zeros((10, 1)),
                cost_returns_to_go=jnp.zeros((10, 1)),
                method=self.policy.loss,
            )
        else:
            # TODO: add MLP acrchitecture here to simplify the code
            raise NotImplementedError

        def get_lr(lr_decay=False):
            if lr_decay is True:
                return optax.cosine_decay_schedule(
                    self.config.lr, decay_steps=self.config.lr_decay_steps
                )
            else:
                return self.config.lr

        def get_optimizer(lr_decay=False, weight_decay=cfg.weight_decay):
            if self.config.max_grad_norm > 0:
                opt = optax.chain(
                    optax.clip_by_global_norm(self.config.max_grad_norm),
                    optax.adamw(get_lr(lr_decay), weight_decay=weight_decay),
                )
            else:
                opt = optax.adamw(get_lr(), weight_decay=weight_decay)
            return opt

        self._train_states["policy"] = TrainState.create(
            params=policy_params,
            tx=get_optimizer(self.config.lr_decay, weight_decay=0.0),
            apply_fn=None,
        )

        self._tgt_params = deepcopy({"policy": policy_params})
        model_keys = ["policy"]

        self._model_keys = tuple(model_keys)

    def get_diff_terms(
        self,
        params,
        observations,
        actions,
        dones,
        observation_conditions,
        env_ts,
        returns_to_go,
        cost_returns_to_go,
        rng,
    ):
        rng, split_rng = jax.random.split(rng)
        ts = jax.random.randint(
            split_rng, dones.shape, minval=0, maxval=self.diffusion.num_timesteps
        )
        rng, split_rng = jax.random.split(rng)
        if self.policy.architecture == "mlp":
            terms = self.policy.apply(
                params["policy"],
                split_rng,
                observations,
                actions,
                observation_conditions,
                ts,
                env_ts=env_ts,
                returns_to_go=returns_to_go,
                cost_returns_to_go=cost_returns_to_go,
                method=self.policy.loss,
            )
        elif self.policy.architecture == "transformer1":
            terms = self.policy.apply(
                params["policy"],
                split_rng,
                observations,
                actions,
                observation_conditions,
                ts,
                env_ts=env_ts,
                returns_to_go=returns_to_go,
                cost_returns_to_go=cost_returns_to_go,
                method=self.policy.loss,
            )
        else:
            raise NotImplementedError

        return terms, ts

    def get_diff_loss(self, batch):
        def diff_loss(params, rng):
            observations = batch["observations"]
            actions = batch["actions"]
            dones = batch["dones"]
            observation_conditions = batch["observation_conditions"]
            env_ts = batch.get("env_ts", None)
            returns_to_go = batch.get("returns_to_go", None)
            cost_returns_to_go = batch.get("cost_returns_to_go", None)
            terms, ts = self.get_diff_terms(
                params,
                observations=observations,
                actions=actions,
                dones=dones,
                observation_conditions=observation_conditions,
                env_ts=env_ts,
                returns_to_go=returns_to_go,
                cost_returns_to_go=cost_returns_to_go,
                rng=rng,
            )
            diff_loss = terms["loss"].mean()

            return diff_loss, terms, ts

        return diff_loss

    @partial(jax.jit, static_argnames=("self", "policy_tgt_update"))
    def _train_step(
        self, train_states, tgt_params, rng, batch, policy_tgt_update=False
    ):
        diff_loss_fn = self.get_diff_loss(batch)

        def policy_loss_fn(params, tgt_params, rng):
            observations = batch["observations"]

            rng, split_rng = jax.random.split(rng)
            diff_loss, _, _ = diff_loss_fn(params, split_rng)

            return (diff_loss,), locals()

        # Calculate policy losses and grads
        params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_policy), grads_policy = value_and_multi_grad(
            policy_loss_fn, 1, has_aux=True
        )(params, tgt_params, rng)

        # Update policy train states
        train_states["policy"] = train_states["policy"].apply_gradients(
            grads=grads_policy[0]["policy"]
        )

        # Update target parameters
        if policy_tgt_update:
            tgt_params["policy"] = update_target_network(
                train_states["policy"].params, tgt_params["policy"], self.config.tau
            )

        metrics = dict(
            diff_loss=aux_policy["diff_loss"],
            policy_grad_norm=optax.global_norm(grads_policy[0]["policy"]),
            policy_weight_norm=optax.global_norm(train_states["policy"].params),
        )

        return train_states, tgt_params, metrics

    def train(self, batch):
        self._total_steps += 1
        policy_tgt_update = (
            self._total_steps > 1000
            and self._total_steps % self.config.policy_tgt_freq == 0
        )
        self._train_states, self._tgt_params, metrics = self._train_step(
            self._train_states, self._tgt_params, next_rng(), batch, policy_tgt_update
        )
        return metrics

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def eval_params(self):
        return {
            key: self.train_states[key].params_ema or self.train_states[key].params
            for key in self.model_keys
        }

    @property
    def total_steps(self):
        return self._total_steps
