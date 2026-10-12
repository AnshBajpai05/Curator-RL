"""Pure-PyTorch stand-in for TRL's Triton log-prob kernel (P100 / no-Triton support).

TRL 1.15 computes GRPO's per-token log-probabilities with `trl.kernels.ChunkedLogProbFunction`, a
Triton kernel. Triton ships with PyTorch on Linux only, and its kernels need a GPU of compute
capability 7.0 or newer. Kaggle's P100 is 6.0, so stock TRL 1.15 GRPO cannot train there (and
cannot run on Windows at all: `_ChunkedLogProbFunction` is None). This module provides the same
interface and the same mathematics in plain PyTorch:

    z = (hidden @ W.T + b) * logit_scale; z = softcap * tanh(z / softcap) if softcapping; z /= T
    log_probs = z[target] - logsumexp(z),  entropy = logsumexp(z) - sum softmax(z) * z,
    log_sum_sq_probs = log sum softmax(z)^2,  mean_logits = mean(z),  is_top1 = target is the argmax

Tokens are processed in chunks under activation checkpointing, so the [N, vocab] logits are never
held for the backward pass (peak extra memory = one chunk's logits). `install_if_needed()` patches
TRL only when the Triton kernel is unavailable or the GPU is pre-Volta; on a T4 TRL's kernel stays.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

TOKEN_CHUNK = 512
OUTPUT_ORDER = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")


def _chunk(hidden, weight, bias, targets, temperature, softcap, logit_scale, want_entropy):
    z = hidden.float() @ weight.float().t()
    if bias is not None:
        z = z + bias.float()
    z = z * logit_scale
    if softcap is not None:
        z = softcap * torch.tanh(z / softcap)
    z = z / temperature
    log_z = torch.logsumexp(z, dim=-1)
    logp = z.gather(1, targets[:, None]).squeeze(1) - log_z
    if not want_entropy:
        return logp, torch.zeros_like(logp)
    p = torch.softmax(z, dim=-1)
    return logp, log_z - (p * z).sum(-1)


class TorchChunkedLogProb:
    """Drop-in for `ChunkedLogProbFunction.apply(...)` (same arguments, same 5-tuple output)."""

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size=None, final_logit_softcapping=None,
              logit_scale=1.0, outputs=OUTPUT_ORDER):
        want = set(outputs)
        logps, ents, lss, means, top1 = [], [], [], [], []
        for s in range(0, hidden.shape[0], TOKEN_CHUNK):
            h, t = hidden[s:s + TOKEN_CHUNK], targets[s:s + TOKEN_CHUNK]
            if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad):
                lp, en = checkpoint(_chunk, h, weight, bias, t, temperature, final_logit_softcapping, logit_scale,
                                    "entropy" in want, use_reentrant=False)
            else:
                lp, en = _chunk(h, weight, bias, t, temperature, final_logit_softcapping, logit_scale,
                                "entropy" in want)
            logps.append(lp)
            ents.append(en)
            if want & {"log_sum_sq_probs", "mean_logits", "is_top1"}:
                with torch.no_grad():
                    z = h.float() @ weight.float().t()
                    if bias is not None:
                        z = z + bias.float()
                    z = z * logit_scale
                    if final_logit_softcapping is not None:
                        z = final_logit_softcapping * torch.tanh(z / final_logit_softcapping)
                    z = z / temperature
                    lss.append(torch.logsumexp(2 * torch.log_softmax(z, -1), -1))
                    means.append(z.mean(-1))
                    top1.append(z.argmax(-1) == t)
        cat = lambda xs: torch.cat(xs) if xs else None  # noqa: E731
        return (cat(logps), cat(ents) if "entropy" in want else None,
                cat(lss) if "log_sum_sq_probs" in want else None,
                cat(means) if "mean_logits" in want else None,
                cat(top1) if "is_top1" in want else None)


def install_if_needed(tu, force: bool = False) -> str:
    """Patch TRL's fused LM head (module `trl.trainer.utils`, passed in by the TRL adapter, the only
    module allowed to import TRL) with the PyTorch path when Triton cannot run. Returns the path used."""
    if tu._ChunkedLogProbFunction is TorchChunkedLogProb:      # already patched in this process
        return "pytorch-fallback"
    pre_volta = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 7
    if force or tu._ChunkedLogProbFunction is None or pre_volta:
        tu._ChunkedLogProbFunction = TorchChunkedLogProb
        return "pytorch-fallback"
    return "triton"
