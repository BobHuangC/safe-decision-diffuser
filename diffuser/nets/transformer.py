# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR observation_conditions OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import jax
import distrax

import flax.linen as nn
import jax.numpy as jnp
from diffuser.diffusion import GaussianDiffusion, ModelMeanType, _extract_into_tensor
from utilities.jax_utils import extend_and_repeat

from diffuser.dpm_solver import DPM_Solver, NoiseScheduleVP
from diffuser.nets.attention import BasicTransformerBlock
from diffuser.nets.helpers import TimeEmbedding


class TransformerCondPolicyNet(nn.Module):
    r"""
    Parameters:
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_linear_projection (`bool`, defaults to `False`): tbd
        only_cross_attention (`bool`, defaults to `False`): tbd
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """

    action_dim: int
    n_heads: int
    depth: int = 1
    embedding_dim: int = 128
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = True
    split_head_dim: bool = False
    max_traj_length: int = 1000
    returns_condition: bool = True
    cost_returns_condition: bool = True
    env_ts_condition: bool = True
    condition_dropout: float = 0.25

    def setup(self):
        inner_dim = self.n_heads * self.embedding_dim
        self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
        self.time_encoder = TimeEmbedding(embed_size=self.embedding_dim, use_mlp=False)
        self.transformer_blocks = [
            BasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.embedding_dim,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
            )
            for _ in range(self.depth)
        ]
        self.returns_encoder = nn.Dense(self.embedding_dim)
        self.cost_returns_encoder = nn.Dense(self.embedding_dim)

        self.proj_out = nn.Dense(self.embedding_dim, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    @nn.compact
    def __call__(
        self,
        observations,
        rng,
        actions: jnp.ndarray,
        timesteps: jnp.ndarray,
        env_ts: jnp.ndarray,
        returns_to_go: jnp.ndarray = None,
        cost_returns_to_go: jnp.ndarray = None,
        use_dropout: bool = True,
        reward_returns_force_dropout: bool = False,
        cost_returns_force_droupout: bool = False,
        context: jnp.ndarray = None,
        deterministic: bool = False,
    ):
        # first to construct the embedding
        # the embedding constructed as time_embed | env_ts_emb | returns_emb | cost_returns_emb | obs_emb | act_emb
        mask_dist = None
        if self.returns_condition or self.cost_returns_condition:
            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)

        env_ts_embed = nn.Embed(self.max_traj_length, self.embedding_dim)(env_ts)
        time_embed = self.time_encoder(timesteps)
        input_embed = [env_ts_embed, time_embed]

        if self.returns_condition:
            assert returns_to_go is not None
            returns_embed = nn.Dense(self.embedding_dim)(returns_to_go.reshape(-1, 1))
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask
            if reward_returns_force_dropout:
                returns_embed = returns_embed * 0
            input_embed.append(returns_embed)

        if self.cost_returns_condition:
            assert cost_returns_to_go is not None
            cost_returns_embed = nn.Dense(self.embedding_dim)(
                cost_returns_to_go.reshape(-1, 1)
            )
            if use_dropout:
                if not self.returns_condition:
                    rng, sample_key = jax.random.split(rng)
                    mask = mask_dist.sample(
                        seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                    )
                cost_returns_embed = cost_returns_embed * mask
            if cost_returns_force_droupout:
                cost_returns_embed = cost_returns_embed * 0
            input_embed.append(cost_returns_embed)

        obs_embed = nn.Dense(self.embedding_dim)(observations)
        input_embed.append(obs_embed)

        act_embed = nn.Dense(self.embedding_dim)(actions)
        input_embed.append(act_embed)

        input_embed = jnp.stack(input_embed, axis=1)
        input_embed = nn.LayerNorm(epsilon=1e-5)(input_embed)
        input_embed = nn.Dropout(rate=self.dropout, deterministic=deterministic)(
            input_embed
        )

        # Then pass the emb into the TransformerBlock
        residual = input_embed
        embed = self.proj_in(input_embed)

        for transformer_block in self.transformer_blocks:
            embed = transformer_block(embed, context, deterministic=deterministic)
        output_embed = self.proj_out(embed)
        output_embed = output_embed + residual
        # output_embed = nn.LayerNorm(epsilon=1e-5)(output_embed)

        output_act_embed = output_embed[:, -1, :]
        output_actions = nn.Dense(self.action_dim)(output_act_embed)
        return output_actions


# Corresponding to TransformerCondPolicyNet
class DiffusionTransformerPolicy(nn.Module):
    diffusion: GaussianDiffusion
    observation_dim: int
    action_dim: int
    # use_layer_norm: bool = False
    use_dpm: bool = False
    sample_method: str = "ddpm"
    dpm_steps: int = 15
    dpm_t_end: float = 0.001
    env_ts_condition: bool = False
    returns_condition: bool = False
    cost_returns_condition: bool = False
    condition_dropout: float = 0.25
    n_heads: int = 4
    depth: int = 1
    dropout: float = 0.0
    embedding_dim: int = 128
    max_traj_length: int = 1000
    architecture: str = "transformer"

    def setup(self):
        self.base_net = TransformerCondPolicyNet(
            action_dim=self.action_dim,
            n_heads=self.n_heads,
            embedding_dim=self.embedding_dim,
            depth=self.depth,
            dropout=self.dropout,
            only_cross_attention=False,
            dtype=jnp.float32,
            use_memory_efficient_attention=False,
            split_head_dim=False,
        )

    def dpm_sample(
        self,
        rng,
        observations,
        observation_conditions,
        env_ts=None,
        deterministic=False,
        returns_to_go=None,
        cost_returns_to_go=None,
        repeat=None,
    ):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        noise_clip = True

        shape = observations.shape[:-1] + (self.action_dim,)

        ns = NoiseScheduleVP(
            schedule="discrete", alphas_cumprod=self.diffusion.alphas_cumprod
        )

        def wrap_model(model_fn, env_ts, returns_to_go, cost_returns_to_go):
            def wrapped_model_fn(x, t):
                t = (t - 1.0 / ns.total_N) * ns.total_N

                out = model_fn(
                    rng,
                    x,
                    t,
                    env_ts=env_ts,
                    returns_to_go=returns_to_go,
                    cost_returns_to_go=cost_returns_to_go,
                )
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
            model_fn=wrap_model(
                partial(self.base_net, observations),
                env_ts=env_ts,
                returns_to_go=returns_to_go,
                cost_returns_to_go=cost_returns_to_go,
            ),
            # observation_conditions=observation_conditions,
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
        observation_conditions,
        env_ts=None,
        deterministic=False,
        returns_to_go=None,
        cost_returns_to_go=None,
        repeat=None,
    ):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)

        shape = observations.shape[:-1] + (self.action_dim,)

        return self.diffusion.ddim_sample_loop(
            rng_key=rng,
            model_forward=partial(self.base_net, observations),
            shape=shape,
            conditions=observation_conditions,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            env_ts=env_ts,
            clip_denoised=True,
        )

    def ddpm_sample(
        self,
        rng,
        observations,
        observation_conditions,
        env_ts=None,
        deterministic=False,
        returns_to_go=None,
        cost_returns_to_go=None,
        repeat=None,
    ):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)

        # shape: (..., action_dim)
        shape = observations.shape[:-1] + (self.action_dim,)
        return self.diffusion.p_sample_loop_jit(
            rng_key=rng,
            model_forward=self.base_net,
            observations=observations,
            shape=shape,
            conditions=observation_conditions,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            env_ts=env_ts,
            clip_denoised=True,
            deterministic=deterministic,
        )

    def __call__(
        self,
        rng,
        observations,
        observation_conditions,
        env_ts=None,
        deterministic=False,
        returns_to_go=None,
        cost_returns_to_go=None,
        repeat=None,
    ):
        return getattr(self, f"{self.sample_method}_sample")(
            rng,
            observations,
            observation_conditions,
            env_ts,
            deterministic,
            returns_to_go,
            cost_returns_to_go,
            repeat,
        )

    def loss(
        self,
        rng_key,
        observations,
        actions,
        observation_conditions,
        ts,
        env_ts=None,
        returns_to_go=None,
        cost_returns_to_go=None,
    ):
        terms = self.diffusion.training_losses(
            rng_key,
            model_forward=partial(self.base_net, observations),
            x_start=actions,
            conditions=observation_conditions,
            t=ts,
            env_ts=env_ts,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
        )
        return terms

    def max_action(self):
        return self.diffusion.max_value
