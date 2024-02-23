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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import math

import flax.linen as nn
import jax
import jax.numpy as jnp


def _query_chunk_attention(query, key, value, precision, key_chunk_size: int = 4096):
    """Multi-head dot product attention with a limited number of queries."""
    num_kv, num_heads, k_features = key.shape[-3:]
    v_features = value.shape[-1]
    key_chunk_size = min(key_chunk_size, num_kv)
    query = query / jnp.sqrt(k_features)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value):
        attn_weights = jnp.einsum(
            "...qhd,...khd->...qhk", query, key, precision=precision
        )

        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)

        exp_values = jnp.einsum(
            "...vhf,...qhv->...qhf", value, exp_weights, precision=precision
        )
        max_score = jnp.einsum("...qhk->...qh", max_score)

        return (exp_values, exp_weights.sum(axis=-1), max_score)

    def chunk_scanner(chunk_idx):
        # julienne key array
        key_chunk = jax.lax.dynamic_slice(
            operand=key,
            start_indices=[0] * (key.ndim - 3) + [chunk_idx, 0, 0],  # [...,k,h,d]
            slice_sizes=list(key.shape[:-3])
            + [key_chunk_size, num_heads, k_features],  # [...,k,h,d]
        )

        # julienne value array
        value_chunk = jax.lax.dynamic_slice(
            operand=value,
            start_indices=[0] * (value.ndim - 3) + [chunk_idx, 0, 0],  # [...,v,h,d]
            slice_sizes=list(value.shape[:-3])
            + [key_chunk_size, num_heads, v_features],  # [...,v,h,d]
        )

        return summarize_chunk(query, key_chunk, value_chunk)

    chunk_values, chunk_weights, chunk_max = jax.lax.map(
        f=chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size)
    )

    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)

    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)

    return all_values / all_weights


def jax_memory_efficient_attention(
    query,
    key,
    value,
    precision=jax.lax.Precision.HIGHEST,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
):
    r"""
    Flax Memory-efficient multi-head dot product attention. https://arxiv.org/abs/2112.05682v2
    https://github.com/AminRezaei0x443/memory-efficient-attention

    Args:
        query (`jnp.ndarray`): (batch..., query_length, head, query_key_depth_per_head)
        key (`jnp.ndarray`): (batch..., key_value_length, head, query_key_depth_per_head)
        value (`jnp.ndarray`): (batch..., key_value_length, head, value_depth_per_head)
        precision (`jax.lax.Precision`, *optional*, defaults to `jax.lax.Precision.HIGHEST`):
            numerical precision for computation
        query_chunk_size (`int`, *optional*, defaults to 1024):
            chunk size to divide query array value must divide query_length equally without remainder
        key_chunk_size (`int`, *optional*, defaults to 4096):
            chunk size to divide key and value array value must divide key_value_length equally without remainder

    Returns:
        (`jnp.ndarray`) with shape of (batch..., query_length, head, value_depth_per_head)
    """
    num_q, num_heads, q_features = query.shape[-3:]

    def chunk_scanner(chunk_idx, _):
        # julienne query array
        query_chunk = jax.lax.dynamic_slice(
            operand=query,
            start_indices=([0] * (query.ndim - 3)) + [chunk_idx, 0, 0],  # [...,q,h,d]
            slice_sizes=list(query.shape[:-3])
            + [min(query_chunk_size, num_q), num_heads, q_features],  # [...,q,h,d]
        )

        return (
            chunk_idx + query_chunk_size,  # unused ignore it
            _query_chunk_attention(
                query=query_chunk,
                key=key,
                value=value,
                precision=precision,
                key_chunk_size=key_chunk_size,
            ),
        )

    _, res = jax.lax.scan(
        f=chunk_scanner,
        init=0,
        xs=None,
        length=math.ceil(num_q / query_chunk_size),  # start counter  # stop counter
    )

    return jnp.concatenate(res, axis=-3)  # fuse the chunked result back


class Attention(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762

    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_q")
        self.key = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_k")
        self.value = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, name="to_v")

        self.proj_attn = nn.Dense(self.query_dim, dtype=self.dtype, name="to_out_0")
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(
        self, hidden_states, context=None, attention_mask=None, deterministic=True
    ):
        context = hidden_states if context is None else context

        query_proj = self.query(hidden_states)
        key_proj = self.key(context)
        value_proj = self.value(context)

        if self.split_head_dim:
            b = hidden_states.shape[0]
            query_states = jnp.reshape(query_proj, (b, -1, self.heads, self.dim_head))
            key_states = jnp.reshape(key_proj, (b, -1, self.heads, self.dim_head))
            value_states = jnp.reshape(value_proj, (b, -1, self.heads, self.dim_head))
        else:
            query_states = self.reshape_heads_to_batch_dim(query_proj)
            key_states = self.reshape_heads_to_batch_dim(key_proj)
            value_states = self.reshape_heads_to_batch_dim(value_proj)

        if self.use_memory_efficient_attention:
            query_states = query_states.transpose(1, 0, 2)
            key_states = key_states.transpose(1, 0, 2)
            value_states = value_states.transpose(1, 0, 2)

            # this if statement create a chunk size for each layer of the unet
            # the chunk size is equal to the query_length dimension of the deepest layer of the unet

            flatten_latent_dim = query_states.shape[-3]
            if flatten_latent_dim % 64 == 0:
                query_chunk_size = int(flatten_latent_dim / 64)
            elif flatten_latent_dim % 16 == 0:
                query_chunk_size = int(flatten_latent_dim / 16)
            elif flatten_latent_dim % 4 == 0:
                query_chunk_size = int(flatten_latent_dim / 4)
            else:
                query_chunk_size = int(flatten_latent_dim)

            hidden_states = jax_memory_efficient_attention(
                query_states,
                key_states,
                value_states,
                query_chunk_size=query_chunk_size,
                key_chunk_size=4096 * 4,
            )

            hidden_states = hidden_states.transpose(1, 0, 2)
        else:
            # compute attentions
            if self.split_head_dim:
                attention_scores = jnp.einsum(
                    "b t n h, b f n h -> b n f t", key_states, query_states
                )
            else:
                attention_scores = jnp.einsum(
                    "b i d, b j d->b i j", query_states, key_states
                )

            attention_scores = attention_scores * self.scale
            attention_probs = nn.softmax(
                attention_scores, axis=-1 if self.split_head_dim else 2
            )

            # attend to values
            if self.split_head_dim:
                hidden_states = jnp.einsum(
                    "b n f t, b t n h -> b f n h", attention_probs, value_states
                )
                b = hidden_states.shape[0]
                hidden_states = jnp.reshape(
                    hidden_states, (b, -1, self.heads * self.dim_head)
                )
            else:
                hidden_states = jnp.einsum(
                    "b i j, b j d -> b i d", attention_probs, value_states
                )
                hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        hidden_states = self.proj_attn(hidden_states)
        return self.dropout_layer(hidden_states, deterministic=deterministic)


class StylizationBlock(nn.Module):
    latent_dim: int
    time_embed_dim: int
    dropout: bool

    @nn.compact
    def __call__(self, h, emb):
        emb_out = nn.Dense(2 * self.latent_dim)(nn.activation.silu(emb))
        emb_out = jnp.expand_dims(emb_out, axis=1)
        scale, shift = emb_out[..., : self.latent_dim], emb_out[..., self.latent_dim :]
        h = nn.LayerNorm()(h) * (1 + scale) + shift
        # TODO(zbzhu): zero initilize this Dense layer
        h = nn.Dense(self.latent_dim)(
            nn.Dropout(rate=self.dropout)(nn.activation.silu(h)),
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )


class BasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762


    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        only_cross_attention (`bool`, defaults to `False`):
            Whether to only apply cross attention.
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        split_head_dim (`bool`, *optional*, defaults to `False`):
            Whether to split the head dimension into a new axis for the self-attention computation. In most cases,
            enabling this flag should speed up the computation for Stable Diffusion 2.x and Stable Diffusion XL.
    """

    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    only_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    time_embed_dim: int = None

    def setup(self):
        # self attention (or cross_attention if only_cross_attention is True)
        self.attn1 = Attention(
            self.dim,
            self.n_heads,
            self.d_head,
            self.dropout,
            self.use_memory_efficient_attention,
            self.split_head_dim,
            dtype=self.dtype,
        )
        # cross attention
        self.attn2 = Attention(
            self.dim,
            self.n_heads,
            self.d_head,
            self.dropout,
            self.use_memory_efficient_attention,
            self.split_head_dim,
            dtype=self.dtype,
        )
        self.ff = FeedForward(dim=self.dim, dropout=self.dropout, dtype=self.dtype)
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        if self.time_embed_dim is not None:
            self.stylization_block1 = StylizationBlock(
                self.dim, self.time_embed_dim, self.dropout
            )
            self.stylization_block2 = StylizationBlock(
                self.dim, self.time_embed_dim, self.dropout
            )
            self.stylization_block3 = StylizationBlock(
                self.dim, self.time_embed_dim, self.dropout
            )

    def __call__(self, hidden_states, time_embed, context, deterministic=True):
        # self attention
        residual = hidden_states
        if self.only_cross_attention:
            assert context is not None
            hidden_states = self.attn1(
                self.norm1(hidden_states), context, deterministic=deterministic
            )
        else:
            hidden_states = self.attn1(
                self.norm1(hidden_states), deterministic=deterministic
            )
        if self.time_embed_dim is not None:
            hidden_states = (
                self.stylization_block1(hidden_states, time_embed) + residual
            )
        else:
            hidden_states = hidden_states + residual

        # cross attention if context is not None
        residual = hidden_states
        hidden_states = self.attn2(
            self.norm2(hidden_states), context, deterministic=deterministic
        )
        if self.time_embed_dim is not None:
            hidden_states = (
                self.stylization_block2(hidden_states, time_embed) + residual
            )
        else:
            hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states), deterministic=deterministic)
        if self.time_embed_dim is not None:
            hidden_states = (
                self.stylization_block3(hidden_states, time_embed) + residual
            )
        else:
            hidden_states = hidden_states + residual

        return self.dropout_layer(hidden_states, deterministic=deterministic)


class FeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a non-linearity. It is the counterpart of PyTorch's
    [`FeedForward`] class, with the following simplifications:
    - The activation function is currently hardcoded to a gated linear unit from:
    https://arxiv.org/abs/2002.05202
    - `dim_out` is equal to `dim`.
    - The number of hidden dimensions is hardcoded to `dim * 4` in [`FlaxGELU`].

    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = GEGLU(self.dim, self.dropout, self.dtype)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states, deterministic=deterministic)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.

    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """

    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype)
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        return self.dropout_layer(
            hidden_linear * nn.gelu(hidden_gelu), deterministic=deterministic
        )
