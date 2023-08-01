import jax
import jax.numpy as jnp

from diffuser.nets.temporal import ResidualTemporalBlock, TemporalUnet


def test_residual_block():
    res_block = ResidualTemporalBlock(out_channels=32, kernel_size=5)
    params = res_block.init(
        jax.random.PRNGKey(0), jnp.ones((1, 3, 16)), jnp.ones((1, 32))
    )
    output = res_block.apply(params, jnp.ones((5, 3, 16)), jnp.ones((5, 32)))
    assert output.shape == (5, 3, 32)


def test_temporal_unet():
    unet = TemporalUnet(transition_dim=4)
    params = unet.init(
        jax.random.PRNGKey(0),
        jax.random.PRNGKey(0),
        jnp.ones((1, 32, 4)),
        jnp.ones((1,)),
    )
    output1 = unet.apply(
        params, jax.random.PRNGKey(0), jnp.ones((5, 32, 4)), jnp.ones((5,))
    )
    output2 = unet.apply(
        params, jax.random.PRNGKey(0), jnp.ones((5, 32, 4)), jnp.ones((5,)), force_dropout=True
    )
    assert output1.shape == output2.shape == (5, 32, 4)
