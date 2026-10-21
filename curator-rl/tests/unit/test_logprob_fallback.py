"""The PyTorch log-prob path equals the naive full-logits computation (values and gradients)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from curator_rl.trainer.logprob_fallback import TorchChunkedLogProb  # noqa: E402


def _naive(hidden, weight, bias, targets, temperature, softcap, scale):
    z = (hidden @ weight.t() + bias) * scale
    if softcap is not None:
        z = softcap * torch.tanh(z / softcap)
    z = z / temperature
    logp_all = torch.log_softmax(z, -1)
    p = logp_all.exp()
    return (logp_all.gather(1, targets[:, None]).squeeze(1), -(p * logp_all).sum(-1),
            torch.logsumexp(2 * logp_all, -1), z.mean(-1), z.argmax(-1) == targets)


@pytest.mark.parametrize("softcap", [None, 30.0])
def test_values_and_gradients_match_the_naive_computation(softcap):
    g = torch.Generator().manual_seed(0)
    n, h, v = 1100, 16, 97                                  # > 2 token chunks of 512
    hidden = torch.randn(n, h, generator=g, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(v, h, generator=g, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(v, generator=g, dtype=torch.float64)
    targets = torch.randint(0, v, (n,), generator=g)
    got = TorchChunkedLogProb.apply(hidden, weight, bias, targets, 0.7, 32768, softcap, 1.3,
                                    ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1"))
    ref = _naive(hidden, weight, bias, targets, 0.7, softcap, 1.3)
    for a, b in zip(got, ref):
        if a.dtype == torch.bool:
            assert torch.equal(a, b)
        else:
            assert torch.allclose(a.double(), b.double(), rtol=1e-5, atol=1e-5)   # fp32 path vs fp64 reference
    loss_got = -(got[0].sum()) + 0.1 * got[1].sum()
    gh, gw = torch.autograd.grad(loss_got, (hidden, weight))
    loss_ref = -(ref[0].sum()) + 0.1 * ref[1].sum()
    rh, rw = torch.autograd.grad(loss_ref, (hidden, weight))
    assert torch.allclose(gh, rh, rtol=1e-4, atol=1e-4) and torch.allclose(gw, rw, rtol=1e-4, atol=1e-4)


def test_unrequested_outputs_are_none():
    hidden, weight = torch.randn(5, 4), torch.randn(11, 4)
    out = TorchChunkedLogProb.apply(hidden, weight, None, torch.zeros(5, dtype=torch.long), 1.0, 1024, None, 1.0, ())
    assert out[0].shape == (5,) and all(x is None for x in out[1:])
