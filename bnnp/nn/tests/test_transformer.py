import torch

from bnnp.nn.transformer import Decoder


def test_decoder_fwd_matches_predict():
    decoder = Decoder(
        dim=128,
        mlp_dim=256,
        head_dim=64,
        depth=2,
        window_size=16,
        use_rope=True,
        rotary_min_freq=0.0001,
        rotary_max_freq=1.0,
    ).cuda()
    decoder.compile(fullgraph=True, dynamic=False)

    # Un-zero-init weights for testing.
    for block in decoder.blocks:
        block["attn"].o_proj.weight.data.normal_(0, 128**-0.5)
        block["mlp"].down_proj.weight.data.normal_(0, 256**-0.5)

    N, T, D = 2, 32, 128
    decoder.precompute_caches(T, "cuda", torch.float32)

    x = torch.randn(N, T, D).to("cuda")
    x_fwd = decoder(x)

    x_pred = []
    cache = None
    for t in range(T):
        x_t, cache = decoder.predict(x[:, t, :], cache)
        x_pred.append(x_t)
    x_pred = torch.stack(x_pred, dim=1)

    err = (x_fwd - x_pred).abs().max()
    print(f"Max error between forward and predict: {err}")
    assert torch.allclose(x_fwd, x_pred, atol=1e-6, rtol=1e-3)


if __name__ == "__main__":
    test_decoder_fwd_matches_predict()
