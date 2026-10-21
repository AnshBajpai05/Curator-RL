"""Question-level filtering (D-102): which groups a step trains on, and the batch handed back to TRL."""

from __future__ import annotations

import pytest

from curator_rl.trainer.group_filter import candidates, has_signal, select_groups, split_method


def test_method_names():
    assert split_method("curator_q") == ("curator", True)
    assert split_method("uniform_q") == ("uniform", True)
    assert split_method("curator_nocal") == ("curator_nocal", False)
    assert split_method("_q") == ("_q", False)


def test_candidates_and_group_selection():
    assert candidates(16, 0.0) == 16 and candidates(16, 1.0) == 16 and candidates(16, 1.75) == 28
    assert has_signal([0.0, 1.0, 1.0]) and not has_signal([1.0, 1.0]) and not has_signal([])
    signal = [False, True, False, True, True, False]
    assert select_groups(signal, 2) == [1, 3]                  # the first informative groups, in order
    assert select_groups(signal, 4) == [0, 1, 3, 4]            # then the rest, order kept
    assert select_groups([False, False, False], 2) == [0, 1]


def test_scheduler_factory_reads_the_base_method():
    from curator_rl.trainer.schedulers import make_run_scheduler

    with pytest.raises(ValueError, match="unknown method 'nope'"):
        make_run_scheduler("nope_q", None, ["a"], seed=0, prompts_per_round=8, domain_of={}, nominal_sizes={})


def test_keep_groups_selects_whole_groups_and_renormalises_the_loss():
    torch = pytest.importorskip("torch")
    trl_adapter = pytest.importorskip("curator_rl.trainer.trl_adapter")     # needs the TRL stack
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]])
    out = {"prompt_ids": torch.arange(6).unsqueeze(1), "completion_ids": torch.arange(24).view(6, 4),
           "completion_mask": mask, "advantages": torch.tensor([0.0, 0.0, 1.0, -1.0, 0.5, -0.5]),
           "old_per_token_logps": torch.zeros(6, 4), "importance_sampling_ratio": torch.ones(6, 1),
           "num_items_in_batch": mask.sum()}
    kept = trl_adapter.keep_groups(out, [0, 2], 2, 6)
    assert kept["prompt_ids"].squeeze(1).tolist() == [0, 1, 4, 5]          # whole groups, row order kept
    assert kept["advantages"].tolist() == [0.0, 0.0, 0.5, -0.5]
    assert kept["completion_mask"].shape == (4, 2)                         # trailing masked columns dropped
    assert kept["completion_ids"].tolist() == [[0, 1], [4, 5], [16, 17], [20, 21]]
    assert kept["old_per_token_logps"].shape == (4, 2) and kept["importance_sampling_ratio"].shape == (4, 1)
    assert int(kept["num_items_in_batch"]) == 6                            # the loss is normalised by kept tokens


def test_training_model_is_released_from_a_vllm_compile_hook():
    torch = pytest.importorskip("torch")
    trl_adapter = pytest.importorskip("curator_rl.trainer.trl_adapter")

    class Backbone(torch.nn.Module):
        def forward(self, x):
            return x + 1

    model = torch.nn.Sequential(Backbone())          # built before vLLM, as the training copy is

    def hooked_call(self, *args, **kwargs):          # what vLLM's in-place class decoration installs
        if self.do_not_compile:
            return self.forward(*args, **kwargs)
        raise AssertionError("would compile")

    hooked_call.__module__ = "vllm.compilation.decorators"
    Backbone.__call__ = hooked_call
    with pytest.raises(AttributeError, match="do_not_compile"):
        model(torch.zeros(1))                          # the OLMo-2 crash
    assert trl_adapter.release_from_vllm_compile(model) == 1
    assert model(torch.zeros(1)).item() == 1.0
    assert trl_adapter.release_from_vllm_compile(model) == 0          # idempotent


def test_clean_uniform_is_an_equal_mix_of_the_useful_arms():
    from curator_rl.trainer.schedulers import make_run_scheduler

    arms = ["gsm8k", "math35", "mbpp", "countdown", "noisy"]
    sched = make_run_scheduler("clean_uniform", None, arms, seed=0, prompts_per_round=32, domain_of={},
                               nominal_sizes={})
    w = sched.select_mixture(None)
    assert w["countdown"] == w["noisy"] == 0.0
    assert w["gsm8k"] == w["math35"] == w["mbpp"] == pytest.approx(1 / 3)
