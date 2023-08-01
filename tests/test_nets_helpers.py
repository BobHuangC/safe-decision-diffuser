import jax
import jax.numpy as jnp

from diffuser.nets.helpers import Conv1dBlock, DownSample1d, UpSample1d


def test_conv1d_block():
    conv = Conv1dBlock(out_channels=32, kernel_size=3)
    params = conv.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 10)))
    output = conv.apply(params, jnp.ones((5, 3, 10)))
    assert output.shape == (5, 3, 32)


def test_downsample1d():
    downsample = DownSample1d(dim=10)
    params = downsample.init(jax.random.PRNGKey(0), jnp.ones((1, 8, 10)))
    output = downsample.apply(params, jnp.ones((5, 8, 10)))
    assert output.shape == (5, 4, 10)


def test_upsample1d():
    upsample = UpSample1d(dim=10)
    params = upsample.init(jax.random.PRNGKey(0), jnp.ones((1, 4, 10)))
    output = upsample.apply(params, jnp.ones((5, 4, 10)))
    assert output.shape == (5, 8, 10)
