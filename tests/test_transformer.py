import jax

from diffuser.nets import TransformerTemporalModel


def test_transformer_1d():
    feat_in = 32
    horizon = 10
    x = jax.random.normal(jax.random.PRNGKey(0), (10, horizon, feat_in))

    model = TransformerTemporalModel(
        in_channels=feat_in,
        n_heads=4,
        d_head=32,
        depth=2,
    )
    params = model.init(jax.random.PRNGKey(0), x, None)
    output = model.apply(params, x, None)

    print(output.shape)


if __name__ == "__main__":
    test_transformer_1d()
