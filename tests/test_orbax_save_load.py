import os
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import orbax
import pytest
from flax.training import orbax_utils
from flax.training.train_state import TrainState


class MLP(nn.Module):
    output_dim: int
    hidden_dims: Tuple[int] = (64, 64)

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.hidden_dims)):
            x = nn.Dense(self.hidden_dims[i])(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


@pytest.fixture(scope="session")
def model_save_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data")
    return fn


def test_orbax_save(model_save_dir):
    print(model_save_dir)
    mlp = MLP(2, (64, 64))

    train_states = {}
    params = mlp.init(jax.random.PRNGKey(0), jnp.ones((1, 2)))
    train_states["mlp"] = TrainState.create(
        params=params,
        tx=optax.adam(1e-3),
        apply_fn=None,
    )

    save_data = {
        "agent_states": train_states,
        "epoch": 0,
    }
    save_dir = os.path.join(model_save_dir, "mlp_model")

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(save_data)
    orbax_checkpointer.save(save_dir, save_data, save_args=save_args, force=True)


def test_orbax_load(model_save_dir):
    save_dir = os.path.join(model_save_dir, "mlp_model")

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(save_dir)

    mlp = MLP(2, (64, 64))
    output = mlp.apply(restored["agent_states"]["mlp"].params, jnp.ones((1, 2)))
    print(output)
