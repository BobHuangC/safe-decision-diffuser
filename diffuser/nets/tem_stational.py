from functools import partial
from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import repeat

from diffuser.diffusion import GaussianDiffusion, ModelMeanType, _extract_into_tensor
from diffuser.dpm_solver import DPM_Solver, NoiseScheduleVP
from diffuser.nets.helpers import TimeEmbedding, mish, multiple_action_q_function
from utilities.jax_utils import extend_and_repeat


class PolicyNet(nn.Module):
    output_dim: int
    dim: int = 128
    arch: Tuple = (256, 256, 256)
    time_embed_size: int = 16
    act: callable = mish
    use_layer_norm: bool = False
    returns_condition: bool = True
    cost_returns_condition: bool = True
    condition_dropout: float = 0.1
    max_traj_length: int = 1000

    
    @nn.compact
    def __call__(
        self, 
        state, 
        rng, 
        action, 
        time: jnp.ndarray, 
        env_ts: jnp.ndarray, 
        returns_to_go: jnp.ndarray,
        cost_returns_to_go: jnp.ndarray,
        use_dropout: bool = True, 
        force_dropout: bool = False
    ):
        act_fn = mish

        if len(time.shape) < len(action.shape) - 1:
            time = repeat(time, "b -> b n", n=action.shape[1])
        time_embed = TimeEmbedding(self.time_embed_size, self.act)(time)
        emb = jnp.concatenate([state, action, time_embed], axis=-1)
        env_ts = jnp.squeeze(env_ts)
        
        env_ts_emb = nn.Embed(self.max_traj_length, self.dim)(env_ts)
        emb = jnp.concatenate([emb, env_ts_emb], axis=-1)


        mask_dist = None
        if self.returns_condition:
            returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.dim),
                    act_fn,
                    nn.Dense(self.dim * 4),
                    act_fn,
                    nn.Dense(self.dim),
                ]
            )
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)

        if self.cost_returns_condition:
            assert self.returns_condition is True
            cost_returns_mlp = nn.Sequential(
                [
                    nn.Dense(self.dim),
                    act_fn,
                    nn.Dense(self.dim * 4),
                    act_fn,
                    nn.Dense(self.dim),
                ]
            )

        if self.returns_condition:
            assert returns_to_go is not None
            returns_to_go = jnp.expand_dims(returns_to_go, axis=-1) 
            returns_embed = returns_mlp(returns_to_go)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=returns_embed.shape
                )
                returns_embed = returns_embed * mask
            if force_dropout:
                returns_embed = returns_embed * 0

            emb = jnp.concatenate([emb, returns_embed], axis = -1)


        if self.cost_returns_condition:
            assert cost_returns_to_go is not None
            cost_returns = jnp.expand_dims(cost_returns_to_go, axis=-1)
            cost_returns_embed = cost_returns_mlp(cost_returns)
            if use_dropout:
                cost_returns_embed = cost_returns_embed * mask
            if force_dropout:
                cost_returns_embed = cost_returns_embed * 0
            emb = jnp.concatenate([emb, cost_returns_embed], axis = -1)



        emb = nn.LayerNorm()(emb)
        # emb = emb.reshape(-1, emb.shape[1] * emb.shape[2])
        x = emb


        for feat in self.arch:
            x = nn.Dense(feat)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.act(x)
        x = nn.Dense(self.output_dim)(x)
        return x
        



class DiffusionPolicy(nn.Module):
    diffusion: GaussianDiffusion
    observation_dim: int
    action_dim: int
    arch: Tuple = (256, 256, 256)
    time_embed_size: int = 16
    act: callable = mish
    use_layer_norm: bool = False
    use_dpm: bool = False
    sample_method: str = "ddpm"
    dpm_steps: int = 15
    dpm_t_end: float = 0.001

    def setup(self):
        self.base_net = PolicyNet(
            output_dim=self.action_dim,
            arch=self.arch,
            time_embed_size=self.time_embed_size,
            act=self.act,
            use_layer_norm=self.use_layer_norm,
        )

    def __call__(
        self, 
        rng, 
        observations, 
        conditions, 
        env_ts, 
        deterministic=False, 
        repeat=None,
        returns_to_go=None,
        cost_returns_to_go=None):
        return getattr(self, f"{self.sample_method}_sample")(
            rng, 
            observations, 
            conditions, 
            env_ts, 
            deterministic, 
            repeat,
            returns_to_go,
            cost_returns_to_go
        )

    def ddpm_sample(
        self, 
        rng, 
        observations, 
        conditions, 
        env_ts, 
        deterministic=False, 
        repeat=None,
        returns_to_go=None,
        cost_returns_to_go=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
            env_ts = extend_and_repeat(env_ts, 1, repeat)
            returns_to_go = extend_and_repeat(returns_to_go, 1, repeat)
            cost_returns_to_go = extend_and_repeat(cost_returns_to_go, 1, repeat)
            

        shape = observations.shape[:-1] + (self.action_dim,)

        return self.diffusion.p_sample_loop(
            rng_key=rng,
            model_forward=partial(self.base_net, observations),
            shape=shape,
            conditions=conditions,
            env_ts=env_ts,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            clip_denoised=True,
        )

    def dpm_sample(
        self, 
        rng, 
        observations, 
        conditions, 
        env_ts, 
        deterministic=False, 
        repeat=None,
        returns_to_go=None,
        cost_returns_to_go=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
            env_ts = extend_and_repeat(env_ts, 1, repeat)
            returns_to_go = extend_and_repeat(returns_to_go, 1, repeat)
            cost_returns_to_go = extend_and_repeat(cost_returns_to_go, 1, repeat)
        noise_clip = True

        shape = observations.shape[:-1] + (self.action_dim,)

        ns = NoiseScheduleVP(
            schedule="discrete", alphas_cumprod=self.diffusion.alphas_cumprod
        )

        def wrap_model(model_fn):
            def wrapped_model_fn(x, t):
                t = (t - 1.0 / ns.total_N) * ns.total_N

                out = model_fn(x, t)
                # add noise clipping
                if noise_clip:
                    t = t.astype(jnp.int32)
                    x_w = _extract_into_tensor(
                        self.diffusion.sqrt_recip_alphas_cumprod, t, x.shape
                    )
                    e_w = _extract_into_tensor(
                        self.diffusion.sqrt_recipm1_alphas_cumprod, t, x.shape
                    )
                    max_value = (self.diffusion.max_value + x_w * x) / e_w
                    min_value = (self.diffusion.min_value + x_w * x) / e_w

                    out = out.clip(min_value, max_value)
                return out

            return wrapped_model_fn

        dpm_sampler = DPM_Solver(
            model_fn=wrap_model(partial(self.base_net, observations)),
            noise_schedule=ns,
            predict_x0=self.diffusion.model_mean_type is ModelMeanType.START_X,
        )
        x = jax.random.normal(rng, shape)
        out = dpm_sampler.sample(x, steps=self.dpm_steps, t_end=self.dpm_t_end)

        return out

    def ddim_sample(
        self, 
        rng, 
        observations, 
        conditions, 
        env_ts, 
        deterministic=False, 
        repeat=None,
        returns_to_go=None,
        cost_returns_to_go=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
            env_ts = extend_and_repeat(env_ts, 1, repeat)
            returns_to_go = extend_and_repeat(returns_to_go, 1, repeat)
            cost_returns_to_go = extend_and_repeat(cost_returns_to_go, 1, repeat)

        shape = observations.shape[:-1] + (self.action_dim,)

        return self.diffusion.ddim_sample_loop(
            rng_key=rng,
            model_forward=partial(self.base_net, observations),
            shape=shape,
            conditions=conditions,
            clip_denoised=True,
            returns=returns_to_go,
            cost_returns=cost_returns_to_go,
        )

    def loss(
        self, 
        rng_key, 
        observations, 
        actions, 
        conditions, 
        ts, 
        env_ts,
        returns_to_go=None,
        cost_returns_to_go=None,
    ):
        terms = self.diffusion.training_losses(
            rng_key,
            model_forward=partial(self.base_net, observations),
            x_start=actions,
            conditions=conditions,
            t=ts,
            env_ts=env_ts,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
        )
        return terms

    @property
    def max_action(self):
        return self.diffusion.max_value


class Critic(nn.Module):
    observation_dim: int
    action_dim: int
    arch: Tuple = (256, 256, 256)
    act: callable = mish
    use_layer_norm: bool = False
    orthogonal_init: bool = False

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)

        for feat in self.arch:
            if self.orthogonal_init:
                x = nn.Dense(
                    feat,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros,
                )(x)
            else:
                x = nn.Dense(feat)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.act(x)

        if self.orthogonal_init:
            x = nn.Dense(
                1,
                kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        else:
            x = nn.Dense(1)(x)
        return jnp.squeeze(x, -1)

    @property
    def input_size(self):
        return self.observation_dim + self.action_dim


class Value(nn.Module):
    observation_dim: int
    arch: Tuple = (256, 256, 256)
    act: callable = mish
    use_layer_norm: bool = False
    orthogonal_init: bool = False

    @nn.compact
    def __call__(self, observations):
        x = observations

        for feat in self.arch:
            if self.orthogonal_init:
                x = nn.Dense(
                    feat,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros,
                )(x)
            else:
                x = nn.Dense(feat)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.act(x)

        if self.orthogonal_init:
            x = nn.Dense(
                1,
                kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        else:
            x = nn.Dense(1)(x)
        return jnp.squeeze(x, -1)

    @property
    def input_size(self):
        return self.observation_dim


class InverseDynamic(nn.Module):
    action_dim: int
    hidden_dims: Tuple[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.hidden_dims)):
            x = nn.Dense(self.hidden_dims[i])(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x
