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
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
from diffuser.diffusion import GaussianDiffusion, ModelMeanType, _extract_into_tensor
from utilities.jax_utils import extend_and_repeat


from diffuser.nets.attention import BasicTransformerBlock, StylizationBlock
from diffuser.nets.helpers import TimeEmbedding
from .helpers import mish

class TransformerTemporalModel(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf


    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
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

    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    time_embed_dim: int = None

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        self.proj_in = nn.Dense(inner_dim, dtype=self.dtype)
        self.time_encoder = TimeEmbedding(self.time_embed_dim)

        self.transformer_blocks = [
            BasicTransformerBlock(
                inner_dim,
                self.n_heads,
                self.d_head,
                dropout=self.dropout,
                only_cross_attention=self.only_cross_attention,
                dtype=self.dtype,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                split_head_dim=self.split_head_dim,
                time_embed_dim=self.time_embed_dim,
            )
            for _ in range(self.depth)
        ]

        self.proj_out = nn.Dense(self.in_channels, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        if self.time_embed_dim is not None:
            self.stylization_block = StylizationBlock(self.in_channels, self.time_embed_dim, self.dropout)

    def __call__(
        self, 
        hidden_states, 
        timesteps, 
        env_ts, 
        returns_to_go,
        cost_returns_to_go,
        use_dropout, 
        force_dropout,
        context, 
        deterministic=True):
        batch, length, channels = hidden_states.shape
        residual = hidden_states
        hidden_states = self.proj_in(self.norm(hidden_states))
        time_embed = self.time_encoder(timesteps)
        
        env_ts_emb = nn.Embed(self.max_traj_length, self.dim)(env_ts)
        emb = jnp.stack([time_embed, env_ts_emb], axis=1)

        act_fn = mish
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
            returns_to_go = returns_to_go.reshape(-1, 1)
            returns_embed = returns_mlp(returns_to_go)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_embed.shape[0], 1)
                )
                returns_embed = returns_embed * mask

            if force_dropout:
                returns_embed = returns_embed * 0
            emb = jnp.concatenate([emb, jnp.expand_dims(returns_embed, 1)], axis=1)

        if self.cost_returns_condition:
            assert cost_returns_to_go is not None
            cost_returns = cost_returns_to_go.reshape(-1, 1)
            cost_returns_embed = cost_returns_mlp(cost_returns)
            if use_dropout:
                cost_returns_embed = cost_returns_embed * mask

            if force_dropout:
                cost_returns_embed = cost_returns_embed * 0
            emb = jnp.concatenate([emb, jnp.expand_dims(cost_returns_embed, 1)], axis=1)

        emb = nn.LayerNorm()(emb)
        emb = emb.reshape(-1, emb.shape[1] * emb.shape[2])


        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, emb, context, deterministic=deterministic)

        hidden_states = self.proj_out(hidden_states)

        if self.time_embed_dim is not None:
            hidden_states = self.stylization_block(hidden_states, emb) + residual
        else:
            hidden_states = hidden_states + residual
        return self.dropout_layer(hidden_states, deterministic=deterministic)




# take in the sequence of combination of actions and states and output the action
class DiffusionDTPolicy(nn.Module):
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
    env_ts_condition: bool = False
    returns_condition: bool = False
    cost_returns_condition: bool = False
    condition_dropout: float = 0.25
    max_traj_length: int = 1000
    architecture: str = "transformer"


    def setup(self):
        # TODO : fix the shape of the input and others
        self.base_net = TransformerTemporalModel(
            in_channels=self.observation_dim,
            n_heads=8,
            d_head=6,
            depth = 1,
            dropout = 0.0,
            only_cross_attention=False,
            dtype=jnp.float32,
            use_memory_efficient_attention=False,
            split_head_dim=False,
            time_embed_dim = None
        )


    def ddpm_sample(
        self,
        rng,
        observations,
        observation_conditions,
        action_conditions,
        env_ts=None,
        deterministic=False,
        returns_to_go=None,
        cost_returns_to_go=None,
        repeat=None,
    ):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)

        shape = observations.shape[:-1] + (self.action_dim,)

        return self.diffusion.p_sample_loop(
            rng_key=rng,
            model_forward=partial(self.base_net, observations),
            shape=shape,
            observation_conditions=observation_conditions,
            returns_to_go=returns_to_go,
            cost_returns_to_go=cost_returns_to_go,
            env_ts=env_ts,
            clip_denoised=True,
        )

    def __call__(
        self,
        rng,
        observations,
        observation_conditions,
        action_conditions,
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
            action_conditions,
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
        action_conditions,
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









# take in the sequence of states and output state, 
# but use inverse dynamics to generate the action
# class DiffusionDTPlanner(nn.Module):
#     diffusion: GaussianDiffusion
#     observation_dim: int
#     action_dim: int
#     arch: Tuple = (256, 256, 256)
#     time_embed_size: int = 16
#     act: callable = mish
#     use_layer_norm: bool = False
#     use_dpm: bool = False
#     sample_method: str = "ddpm"
#     dpm_steps: int = 15
#     dpm_t_end: float = 0.001
#     env_ts_condition: bool = False
#     returns_condition: bool = False
#     cost_returns_condition: bool = False
#     condition_dropout: float = 0.25
#     max_traj_length: int = 1000

#     def setup(self):
#         self.base_net = TransformerTemporalModel(
#             in_channels=self.observation_dim,
#             n_heads=8,
#             d_head=6,
#             depth = 1,
#             dropout = 0.0,
#             only_cross_attention=False,
#             dtype=jnp.float32,
#             use_memory_efficient_attention=False,
#             split_head_dim=False,
#             time_embed_dim = None
#         )

#     def ddpm_sample(
#         self,
#         rng,
#         observation_conditions,
#         env_ts,
#         deterministic=False,
#         returns_to_go=None,
#         cost_returns_to_go=None,
#     ):
#         batch_size = list(observation_conditions.values())[0].shape[0]
#         return self.diffusion.p_sample_loop_jit(
#             rng_key=rng,
#             model_forward=self.base_net,
#             shape=(batch_size, self.horizon + self.history_horizon, self.sample_dim),
#             observation_conditions=observation_conditions,
#             condition_dim=self.sample_dim - self.action_dim,
#             returns_to_go=returns_to_go,
#             cost_returns_to_go=cost_returns_to_go,
#             env_ts=env_ts,
#             clip_denoised=True,
#         )

#     def __call__(
#         self,
#         rng,
#         observation_conditions,
#         env_ts,
#         deterministic=False,
#         returns_to_go=None,
#         cost_returns_to_go=None,
#     ):
#         return getattr(self, f"{self.sample_method}_sample")(
#             rng, observation_conditions, env_ts, deterministic, returns_to_go, cost_returns_to_go
#         )

#     def loss(
#         self,
#         rng_key,
#         samples,
#         observation_conditions,
#         ts,
#         env_ts,
#         masks,
#         returns_to_go=None,
#         cost_returns_to_go=None,
#     ):
#         terms = self.diffusion.training_losses(
#             rng_key,
#             model_forward=self.base_net,
#             x_start=samples,
#             observation_conditions=observation_conditions,
#             condition_dim=self.sample_dim - self.action_dim,
#             returns_to_go=returns_to_go,
#             cost_returns_to_go=cost_returns_to_go,
#             env_ts=env_ts,
#             t=ts,
#             masks=masks,
#         )
#         return terms

