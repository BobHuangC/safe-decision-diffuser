from functools import partial

import einops
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from .base_algo import Algo
from diffuser.diffusion import GaussianDiffusion, ModelMeanType
from utilities.jax_utils import next_rng, value_and_multi_grad


class DecisionDiffuser(Algo):
    def __init__(self, cfg, planner, inv_model):
        self.config = cfg
        self.planner = planner
        self.inv_model = inv_model
        self.observation_dim = planner.observation_dim
        self.action_dim = inv_model.action_dim
        self.horizon = self.config.horizon
        self.diffusion: GaussianDiffusion = self.planner.diffusion
        self.diffusion.loss_weights = self.get_loss_weights(self.config.loss_discount)

        self._total_steps = 0
        self._train_states = {}

        def get_lr(lr_decay=False):
            return self.config.lr

        def get_optimizer(lr_decay=False, weight_decay=cfg.weight_decay):
            opt = optax.adam(self.config.lr)
            return opt

        planner_params = self.planner.init(
            next_rng(),
            next_rng(),
            jnp.zeros((10, self.horizon, self.observation_dim)),  # samples
            {0: jnp.zeros((10, self.observation_dim))},  # conditions
            jnp.zeros((10,), dtype=jnp.int32),  # ts
            returns=jnp.zeros((10, 1)),  # returns
            cost_returns=jnp.zeros((10, 1)),  # cost_returns
            method=self.planner.loss,
        )
        self._train_states["planner"] = TrainState.create(
            params=planner_params,
            tx=get_optimizer(self.config.lr_decay, weight_decay=0.0),
            apply_fn=None,
        )

        inv_model_params = self.inv_model.init(
            next_rng(),
            jnp.zeros((10, self.observation_dim * 2)),
        )
        self._train_states["inv_model"] = TrainState.create(
            params=inv_model_params,
            tx=get_optimizer(),
            apply_fn=None,
        )

        model_keys = ["planner", "inv_model"]
        self._model_keys = tuple(model_keys)

    def get_loss_weights(self, discount: float) -> jnp.ndarray:
        dim_weights = np.ones(self.observation_dim, dtype=np.float32)

        # decay loss with trajectory timestep: discount**t
        discounts = discount ** np.arange(self.horizon, dtype=np.float32)
        discounts = discounts / discounts.mean()
        loss_weights = einops.einsum(discounts, dim_weights, "h,t->h t")
        # Cause things are conditioned on t=0
        if self.diffusion.model_mean_type == ModelMeanType.EPSILON:
            loss_weights[0] = 0

        return jnp.array(loss_weights)

    @partial(jax.jit, static_argnames=("self"))
    def _train_step(self, train_states, rng, batch):
        diff_loss_fn = self.get_diff_loss(batch)
        inv_loss_fn = self.get_inv_loss(batch)

        params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_planner), grad_planner = value_and_multi_grad(
            diff_loss_fn, 1, has_aux=True
        )(params, rng)

        params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_inv_model), grad_inv_model = value_and_multi_grad(
            inv_loss_fn, 1, has_aux=True
        )(params, rng)

        train_states["planner"] = train_states["planner"].apply_gradients(
            grads=grad_planner[0]["planner"]
        )
        train_states["inv_model"] = train_states["inv_model"].apply_gradients(
            grads=grad_inv_model[0]["inv_model"]
        )

        metrics = dict(
            diff_loss=aux_planner["loss"],
            inv_loss=aux_inv_model["loss"],
        )

        return train_states, metrics

    def get_inv_loss(self, batch):
        def inv_loss(params, rng):
            samples = batch["samples"]
            actions = batch["actions"]

            samples_t = samples[:, :-1]
            samples_tp1 = samples[:, 1:]
            samples_comb = jnp.concatenate([samples_t, samples_tp1], axis=-1)
            samples_comb = jnp.reshape(samples_comb, (-1, self.observation_dim * 2))

            actions = actions[:, :-1]
            actions = jnp.reshape(actions, (-1, self.action_dim))

            pred_actions = self.inv_model.apply(params["inv_model"], samples_comb)
            loss = jnp.mean((pred_actions - actions) ** 2)
            return (loss,), locals()

        return inv_loss

    def get_diff_loss(self, batch):
        def diff_loss(params, rng):
            samples = batch["samples"]
            conditions = batch["conditions"]
            returns = batch.get("returns", None)
            cost_returns = batch.get("cost_returns", None)
            terms, ts = self.get_diff_terms(
                params, samples, conditions, returns, cost_returns, rng
            )
            loss = terms["loss"].mean()

            return (loss,), locals()

        return diff_loss

    def get_diff_terms(self, params, samples, conditions, returns, cost_returns, rng):
        rng, split_rng = jax.random.split(rng)
        ts = jax.random.randint(
            split_rng,
            (samples.shape[0],),
            minval=0,
            maxval=self.diffusion.num_timesteps,
        )
        rng, split_rng = jax.random.split(rng)
        terms = self.planner.apply(
            params["planner"],
            split_rng,
            samples,
            conditions,
            ts,
            returns=returns,
            cost_returns=cost_returns,
            method=self.planner.loss,
        )

        return terms, ts

    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_step(
            self._train_states, next_rng(), batch
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
    def total_steps(self):
        return self._total_steps
