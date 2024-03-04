from functools import partial

import jax
import jax.numpy as jnp

from utilities.jax_utils import next_rng


class SamplerPolicy(object):  # used for cdbc and dql
    def __init__(
        self, policy, qf=None, mean=0, std=1, num_samples=50, act_method="ddpm"
    ):
        self.policy = policy
        self.qf = qf
        self.mean = mean
        self.std = std
        self.num_samples = num_samples
        self.act_method = act_method

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def act(
        self,
        params,
        rng,
        observations,
        env_ts,
        returns_to_go,
        cost_returns_to_go,
        deterministic,
    ):
        observation_conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            observation_conditions,
            env_ts,
            deterministic,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            repeat=None,
        )

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ensemble_act(
        self,
        params,
        rng,
        observations,
        env_ts,
        deterministic,
        returns_to_go,
        cost_returns_to_go,
        num_samples,
    ):
        rng, key = jax.random.split(rng)
        observation_conditions = {}
        actions = self.policy.apply(
            params["policy"],
            key,
            observations,
            observation_conditions,
            env_ts,
            deterministic,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            repeat=num_samples,
        )
        q1 = self.qf.apply(params["qf1"], observations, actions)
        q2 = self.qf.apply(params["qf2"], observations, actions)
        q = jnp.minimum(q1, q2)

        idx = jax.random.categorical(rng, q)
        return actions[jnp.arange(actions.shape[0]), idx]

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ddpmensemble_act(
        self,
        params,
        rng,
        observations,
        env_ts,
        returns_to_go,
        cost_returns_to_go,
        deterministic,
        num_samples,
    ):
        rng, key = jax.random.split(rng)
        observation_conditions = {}
        actions = self.policy.apply(
            params["policy"],
            rng,
            observations,
            observation_conditions,
            env_ts,
            deterministic,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            method=self.policy.ddpm_sample,
            repeat=num_samples,
        )
        q1 = self.qf.apply(params["qf1"], observations, actions)
        q2 = self.qf.apply(params["qf2"], observations, actions)
        q = jnp.minimum(q1, q2)

        idx = jax.random.categorical(rng, q)
        return actions[jnp.arange(actions.shape[0]), idx]

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def dpmensemble_act(
        self,
        params,
        rng,
        observations,
        env_ts,
        returns_to_go,
        cost_returns_to_go,
        deterministic,
        num_samples,
    ):
        rng, key = jax.random.split(rng)
        observation_conditions = {}
        actions = self.policy.apply(
            params["policy"],
            rng,
            observations,
            observation_conditions,
            env_ts=env_ts,
            deterministic=deterministic,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            method=self.policy.dpm_sample,
            repeat=num_samples,
        )
        q1 = self.qf.apply(params["qf1"], observations, actions)
        q2 = self.qf.apply(params["qf2"], observations, actions)
        q = jnp.minimum(q1, q2)

        idx = jax.random.categorical(rng, q)
        return actions[jnp.arange(actions.shape[0]), idx]

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def dpm_act(
        self,
        params,
        rng,
        observations,
        env_ts,
        returns_to_go,
        cost_returns_to_go,
        deterministic,
        num_samples,
    ):
        observation_conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            observation_conditions,
            env_ts=env_ts,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            method=self.policy.dpm_sample,
        )

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ddim_act(
        self,
        params,
        rng,
        observations,
        env_ts,
        returns_to_go,
        cost_returns_to_go,
        deterministic,
        num_samples,
    ):
        observation_conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            observation_conditions,
            env_ts=env_ts,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            method=self.policy.ddim_sample,
        )

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ddpm_act(
        self,
        params,
        rng,
        observations,
        env_ts,
        returns_to_go,
        cost_returns_to_go,
        deterministic,
        num_samples,
    ):
        observation_conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            observation_conditions,
            env_ts=env_ts,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            method=self.policy.ddpm_sample,
        )

    def __call__(
        self,
        observations,
        env_ts=None,
        returns_to_go=None,
        cost_returns_to_go=None,
        deterministic=False,
    ):
        if len(observations.shape) > 2:
            observations = observations.squeeze(1)
        actions = getattr(self, f"{self.act_method}_act")(
            self.params,
            next_rng(),
            observations,
            env_ts,
            returns_to_go,
            cost_returns_to_go,
            deterministic,
            self.num_samples,
        )
        if isinstance(actions, tuple):
            actions = actions[0]
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)


class DiffuserPolicy(object):
    def __init__(self, planner, inv_model, act_method: str = "ddpm"):
        self.planner = planner
        self.inv_model = inv_model
        self.act_method = act_method

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def ddpm_act(
        self,
        params,
        rng,
        observations,
        env_ts,
        returns_to_go,
        cost_returns_to_go,
        deterministic,
    ):  # deterministic is not used
        history_horizon = self.planner.history_horizon
        observation_conditions = {(0, history_horizon + 1): observations}
        plan_samples = self.planner.apply(
            params["planner"],
            rng,
            conditions=observation_conditions,
            env_ts=env_ts,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            method=self.planner.ddpm_sample,
        )

        if self.inv_model is not None:
            obs_comb = jnp.concatenate(
                [
                    plan_samples[:, history_horizon],
                    plan_samples[:, history_horizon + 1],
                ],
                axis=-1,
            )
            actions = self.inv_model.apply(
                params["inv_model"],
                obs_comb,
            )
        else:
            actions = plan_samples[:, history_horizon, -self.planner.action_dim :]

        return actions

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def ddim_act(
        self,
        params,
        rng,
        observations,
        env_ts,
        returns_to_go,
        cost_returns_to_go,
        deterministic,
    ):  # deterministic is not used
        history_horizon = self.planner.history_horizon
        observation_conditions = {(0, history_horizon + 1): observations}
        plan_samples = self.planner.apply(
            params["planner"],
            rng,
            observation_conditions=observation_conditions,
            env_ts=env_ts,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            method=self.planner.ddim_sample,
        )

        if self.inv_model is not None:
            obs_comb = jnp.concatenate(
                [
                    plan_samples[:, history_horizon],
                    plan_samples[:, history_horizon + 1],
                ],
                axis=-1,
            )
            actions = self.inv_model.apply(
                params["inv_model"],
                obs_comb,
            )
        else:
            actions = plan_samples[:, history_horizon, -self.planner.action_dim :]

        return actions

    def __call__(
        self,
        observations,
        env_ts=None,
        returns_to_go=None,
        cost_returns_to_go=None,
        deterministic=False,
    ):
        actions = getattr(self, f"{self.act_method}_act")(
            self.params,
            next_rng(),
            observations,
            env_ts,
            returns_to_go,
            cost_returns_to_go,
            deterministic,
        )
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)
