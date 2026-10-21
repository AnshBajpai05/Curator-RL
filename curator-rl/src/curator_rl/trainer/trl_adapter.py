"""The only module that imports TRL (architecture rule): CURATOR driving an unmodified GRPO.

How the mixture reaches GRPO without a dataloader lag. TRL's dataloader prefetches batches ahead
of the step that uses them, so a dataset that decided the environment at fetch time would apply a
new mixture one step late (TRL_SPIKE G.2 #4, untested until now). Instead the dataset holds
placeholder rows and `CuratorGRPOTrainer._generate_and_score_completions`, the method TRL calls
right before generating, swaps in this step's prompts from the controller. Everything after that
point (generation, reward, advantage, loss, optimizer) is stock TRL.

Settings pinned deliberately (TRL 1.15 defaults that would bite on a T4 or break resume):
`bf16=False, fp16=<cuda>` (T4 and P100 have no bf16), `lr_scheduler_type="constant"` (protocol:
constant LR; also makes resume exact), `top_k=0, repetition_penalty=1.0` (D-84 sampling bug),
`beta=0` / `loss_type="dapo"` / `scale_rewards="group"` (TRL_SPIKE), `save_strategy="no"` (our
own crash-safe checkpoints), `remove_unused_columns=False` (the reward needs env_id etc.).
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer


class CuratorGRPOTrainer(GRPOTrainer):
    """GRPOTrainer whose generation batches are filled by the CURATOR controller.

    With question filtering (D-102, "_q" methods) a step draws more questions than it trains on,
    generates and scores them all in one stock TRL call, and keeps the groups with a reward signal.
    """

    curator_controller = None

    def _generate_and_score_completions(self, inputs):
        ctl = self.curator_controller
        if not (self.model.training and ctl is not None):
            return super()._generate_and_score_completions(inputs)
        from curator_rl.trainer.group_filter import has_signal, select_groups

        g = self.num_generations
        n_keep = len(inputs) // g
        n_draw = ctl.candidates_per_step(n_keep)
        output = super()._generate_and_score_completions(ctl.fill_generation_batch(list(inputs), g, n_draw))
        if n_draw == n_keep:
            return output
        # advantages are per group: a group without a reward signal has all of them at zero
        adv = output["advantages"].detach().float().view(n_draw, g).cpu().tolist()
        signal = [has_signal(a) for a in adv]
        keep = select_groups(signal, n_keep)
        self._metrics["train"]["curator/signal_groups_drawn"].append(sum(signal) / n_draw)
        self._metrics["train"]["curator/signal_groups_kept"].append(sum(signal[i] for i in keep) / n_keep)
        return keep_groups(output, keep, g, n_draw * g, gather=self.accelerator.gather)


# per-completion tensors whose second dimension is the completion length (right-padded)
_COMPLETION_KEYS = ("completion_ids", "completion_mask", "old_per_token_logps", "ref_per_token_logps",
                    "sampling_per_token_logps", "importance_sampling_ratio", "tool_mask")


def keep_groups(output: dict, keep: list[int], g: int, n_rows: int, gather=None) -> dict:
    """A generation batch restricted to whole groups `keep` (row order kept).

    Completion columns are trimmed to the last token any kept row trains on (trailing masked columns
    change nothing: attention is causal) and the loss normaliser, the number of completion tokens in
    the batch, is recomputed for the kept rows. Everything else per row is selected as it is.
    """
    rows = torch.tensor([i * g + j for i in keep for j in range(g)], dtype=torch.long)
    out: dict = {}
    for key, value in output.items():
        if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == n_rows:
            out[key] = value.index_select(0, rows.to(value.device))
        elif isinstance(value, list) and len(value) == n_rows:
            out[key] = [value[i] for i in rows.tolist()]
        else:
            out[key] = value
    mask = out["completion_mask"]
    full = mask.shape[1]
    used = mask.nonzero()
    width = int(used[:, 1].max().item()) + 1 if used.numel() else 1
    if width < full:
        for key in _COMPLETION_KEYS:
            value = out.get(key)
            if torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == full:
                out[key] = value[:, :width]
    loss_mask = out["completion_mask"] if out.get("tool_mask") is None else out["completion_mask"] * out["tool_mask"]
    total = loss_mask.sum()
    out["num_items_in_batch"] = gather(total).sum() if gather is not None else total
    return out


def release_from_vllm_compile(model) -> int:
    """Let the training model run plain forwards after vLLM decorated its classes in place.

    vLLM's Transformers backend (used for e.g. OLMo-2 by vLLM 0.31; Qwen has a native vLLM implementation)
    wraps the *shared* Hugging Face model class with its torch.compile hook. The training copy, built before
    vLLM, lacks the hook's instance state and crashed on its first forward ("'Olmo2Model' object has no
    attribute 'do_not_compile'"). Marking those modules `do_not_compile` makes the hook run the module's
    own forward, as it does whenever compilation is off. Returns the number of modules marked.
    """
    marked = 0
    for module in model.modules():
        call = next((vars(c)["__call__"] for c in type(module).__mro__ if "__call__" in vars(c)), None)
        if str(getattr(call, "__module__", "")).startswith("vllm.") and "do_not_compile" not in vars(module):
            module.do_not_compile = True
            marked += 1
    return marked


@contextlib.contextmanager
def _vllm_room_for(factor: float):
    """TRL caps vLLM's concurrent sequences at one generation batch; question filtering generates
    `factor` times as many, which would otherwise run in two waves. Widen the cap while the trainer
    (and its vLLM engine) is built; nothing else changes."""
    import trl.trainer.grpo_trainer as grpo

    if factor <= 1.0:
        yield
        return
    original = grpo.VLLMGeneration

    class _Wider(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args, max_num_seqs=None, **kwargs):
            if max_num_seqs is not None:
                max_num_seqs = math.ceil(max_num_seqs * factor)
            super().__init__(*args, max_num_seqs=max_num_seqs, **kwargs)

    grpo.VLLMGeneration = _Wider
    try:
        yield
    finally:
        grpo.VLLMGeneration = original


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CuratorCallback(TrainerCallback):
    """Times optimizer steps, hands them to the controller, checkpoints, stops on budget."""

    def __init__(self, controller, out_dir: Path, resume_dir: Path | None) -> None:
        self.ctl, self.out_dir, self.resume_dir = controller, Path(out_dir), resume_dir
        self.trainer: CuratorGRPOTrainer | None = None
        self._t0 = 0.0

    def on_train_begin(self, args, state, control, **kw):
        if self.resume_dir is not None:
            opt_path = self.resume_dir / "optimizer.pt"
            if opt_path.exists() and kw.get("optimizer") is not None:
                kw["optimizer"].load_state_dict(torch.load(opt_path, map_location="cpu"))
            scaler = getattr(getattr(self.trainer, "accelerator", None), "scaler", None)
            if scaler is not None and (self.resume_dir / "scaler.pt").exists():
                scaler.load_state_dict(torch.load(self.resume_dir / "scaler.pt", map_location="cpu"))

    def on_step_begin(self, args, state, control, **kw):
        _sync()
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        _sync()
        self.ctl.on_step_end(time.perf_counter() - self._t0)
        if self.ctl.checkpoint_due:
            self.ctl.save(lambda d: self.save_backend(d, kw))     # retried, then synced; never raises
            self.ctl.checkpoint_due = False
        if self.ctl.stop:
            control.should_training_stop = True
        return control

    def save_backend(self, ckpt_dir: Path, kw: dict) -> None:
        model = kw.get("model") or self.trainer.model
        model.save_pretrained(ckpt_dir / "adapter")
        if kw.get("optimizer") is not None:
            torch.save(kw["optimizer"].state_dict(), ckpt_dir / "optimizer.pt")
        scaler = getattr(getattr(self.trainer, "accelerator", None), "scaler", None)
        if scaler is not None:
            torch.save(scaler.state_dict(), ckpt_dir / "scaler.pt")


def _load_model(name: str, dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:                                     # older transformers
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


def _render(tok, messages) -> str:
    return tok.apply_chat_template([dict(m) for m in messages], tokenize=False, add_generation_prompt=True)


def _hf_greedy(trainer, tok, batch: int = 16):
    def generate(prompts, cap):
        model = trainer.accelerator.unwrap_model(trainer.model)
        was_training = model.training
        model.eval()
        texts = []
        for i in range(0, len(prompts), batch):
            chunk = [_render(tok, p.messages) for p in prompts[i:i + batch]]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, max_new_tokens=int(cap), repetition_penalty=1.0,
                                     pad_token_id=tok.pad_token_id)
            texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if was_training:
            model.train()
        return texts
    return generate


def _vllm_greedy(trainer, tok):
    from vllm import SamplingParams

    def generate(prompts, cap):
        # the step that just finished updated the weights; push them before evaluating
        if trainer.state.global_step != getattr(trainer, "_last_loaded_step", -1):
            trainer.vllm_generation.sync_weights()
            trainer._last_loaded_step = trainer.state.global_step
        params = SamplingParams(temperature=0.0, max_tokens=int(cap), repetition_penalty=1.0)
        outs = trainer.vllm_generation.llm.generate([_render(tok, p.messages) for p in prompts], params,
                                                    use_tqdm=False)
        return [o.outputs[0].text for o in outs]
    return generate


def _sampling_params(k: int, temperature: float, cap: int, seed: int):
    """Exactly the training sampler (TRL colocate: top_p 1, top_k 0, min_p 0, no repetition penalty), set
    explicitly: vLLM fills unset fields from the model's generation_config (Qwen: top_k 20, top_p 0.8)."""
    from vllm import SamplingParams

    return SamplingParams(n=int(k), temperature=float(temperature), top_p=1.0, top_k=0, min_p=0.0,
                          repetition_penalty=1.0, max_tokens=int(cap), seed=int(seed))


def _vllm_sampled(trainer, tok, temperature: float):
    def generate(prompts, cap, k, seeds):
        if trainer.state.global_step != getattr(trainer, "_last_loaded_step", -1):
            trainer.vllm_generation.sync_weights()
            trainer._last_loaded_step = trainer.state.global_step
        # one seed per item: its own generator, so evaluating never moves the generator training samples from
        params = [_sampling_params(k, temperature, cap, s) for s in seeds]
        outs = trainer.vllm_generation.llm.generate([_render(tok, p.messages) for p in prompts], params,
                                                    use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]
    return generate


def _hf_sampled(get_model, tok, temperature: float, batch: int = 8):
    """HF backend: k samples per item, one seed per chunk of items (approximate; vLLM is exact per item)."""
    def generate(prompts, cap, k, seeds):
        model = get_model()
        was_training = model.training
        model.eval()
        out: list[list[str]] = []
        devices = [model.device] if model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):       # leave the training RNG where it was
            for i in range(0, len(prompts), batch):
                chunk = [_render(tok, p.messages) for p in prompts[i:i + batch]]
                enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                torch.manual_seed(int(seeds[i]))
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=float(temperature), top_p=1.0, top_k=0,
                                         num_return_sequences=int(k), max_new_tokens=int(cap),
                                         repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                out += [texts[j * k:(j + 1) * k] for j in range(len(chunk))]
        if was_training:
            model.train()
        return out
    return generate


def _engine_dtype() -> str:
    """What TRL's colocated vLLM runs in ("auto" on a bf16 checkpoint): bf16 from compute capability 8.0,
    else fp16 (T4). Set explicitly so the base model and merged checkpoints are evaluated the same way."""
    return "bfloat16" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else "float16"


def _merged_checkpoint(run_cfg, ckpt_dir: Path) -> str:
    """Base model + the saved LoRA adapter merged (as TRL merges before loading weights into vLLM), in a
    temporary directory that is removed at exit."""
    import atexit
    import shutil
    import tempfile

    cuda = torch.cuda.is_available()
    model = _load_model(run_cfg.model, torch.float16 if cuda else torch.float32)
    if cuda:
        model = model.to("cuda")
    merged = PeftModel.from_pretrained(model, str(Path(ckpt_dir) / "adapter")).merge_and_unload()
    out = tempfile.mkdtemp(prefix="curator_merged_")
    atexit.register(shutil.rmtree, out, True)
    merged.to("cpu").save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(run_cfg.model).save_pretrained(out)
    del merged, model
    if cuda:
        torch.cuda.empty_cache()
    return out


def _attach_vllm_engine(controller, run_cfg, model_dir: str, tok) -> None:
    """A standalone vLLM engine for evaluation only (no trainer in this process), set up like TRL's."""
    import os

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")     # in this process, as colocate is
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_dir, dtype=_engine_dtype(), gpu_memory_utilization=run_cfg.vllm_gpu_memory_utilization,
              max_num_batched_tokens=4096, seed=0)

    def greedy(prompts, cap):
        params = SamplingParams(temperature=0.0, max_tokens=int(cap), repetition_penalty=1.0)
        outs = llm.generate([_render(tok, p.messages) for p in prompts], params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def sampled(prompts, cap, k, seeds):
        params = [_sampling_params(k, run_cfg.eval_temperature, cap, s) for s in seeds]
        outs = llm.generate([_render(tok, p.messages) for p in prompts], params, use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]

    controller.generate, controller.generate_sampled = greedy, sampled


def attach_eval_generate(controller, run_cfg, ckpt_dir: Path | None, *, backend: str = "hf",
                         batch: int = 16) -> None:
    """Evaluation only: a saved policy (base model + its LoRA adapter), or the untrained base model when
    `ckpt_dir` is None. vLLM backend: the merged weights in a standalone vLLM engine (the engine every
    trained run's evaluation uses); otherwise HF decoding."""
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(run_cfg.model)
    if backend == "vllm":
        model_dir = run_cfg.model if ckpt_dir is None else _merged_checkpoint(run_cfg, Path(ckpt_dir))
        _attach_vllm_engine(controller, run_cfg, model_dir, tok)
        return
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = _load_model(run_cfg.model, torch.float16 if cuda else torch.float32)
    if ckpt_dir is not None:
        model = PeftModel.from_pretrained(model, str(Path(ckpt_dir) / "adapter"))
    model = model.to("cuda" if cuda else "cpu").eval()

    def generate(prompts, cap):
        texts = []
        for i in range(0, len(prompts), batch):
            chunk = [_render(tok, p.messages) for p in prompts[i:i + batch]]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False, max_new_tokens=int(cap), repetition_penalty=1.0,
                                     pad_token_id=tok.pad_token_id)
            texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return texts

    controller.generate = generate
    controller.generate_sampled = _hf_sampled(lambda: model, tok, run_cfg.eval_temperature)


def run_trl(controller, run_cfg, out_dir: Path, *, backend: str, resume_dir: Path | None,
            gradient_checkpointing: bool = True) -> None:
    """Train until the controller stops (budget or max rounds); checkpoints along the way."""
    import trl.trainer.utils as trl_utils

    from curator_rl.trainer.logprob_fallback import install_if_needed

    cuda = torch.cuda.is_available()
    controller.logprob_path = install_if_needed(trl_utils)     # P100 / no Triton: PyTorch log-prob path
    print(f"[curator] GRPO log-prob path: {controller.logprob_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(run_cfg.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = _load_model(run_cfg.model, torch.float16 if cuda else torch.float32)
    peft_config = None
    if resume_dir is not None and (resume_dir / "adapter").exists():
        model = PeftModel.from_pretrained(model, str(resume_dir / "adapter"), is_trainable=True)
    else:
        peft_config = LoraConfig(r=run_cfg.lora_r, lora_alpha=2 * run_cfg.lora_r, target_modules="all-linear",
                                 task_type="CAUSAL_LM")

    g, p = run_cfg.group_size, run_cfg.prompts_per_step
    completions_per_step = g * p
    micro = min(run_cfg.micro_batch, completions_per_step)
    if completions_per_step % micro:
        raise ValueError(f"micro_batch {micro} must divide P x G = {completions_per_step}")
    remaining_steps = max((run_cfg.max_rounds - controller.round) * run_cfg.steps_per_round, 1)
    placeholder = {"prompt": [{"role": "user", "content": "placeholder"}], "env_id": "", "prompt_id": "",
                   "reference": "", "meta_json": "{}"}
    dataset = Dataset.from_list([placeholder] * (p * (remaining_steps + 1)))

    kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir / "trl"), per_device_train_batch_size=micro,
        gradient_accumulation_steps=completions_per_step // micro, num_generations=g,
        max_completion_length=run_cfg.max_completion_tokens, learning_rate=run_cfg.learning_rate,
        lr_scheduler_type="constant", warmup_steps=0, max_steps=remaining_steps, fp16=cuda, bf16=False,
        use_cpu=not cuda, logging_steps=1, report_to="none", save_strategy="no", remove_unused_columns=False,
        seed=run_cfg.seed + 1000 * controller.round, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        beta=0.0, loss_type="dapo", scale_rewards="group", gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=0,
    )
    if backend == "vllm":
        kwargs.update(use_vllm=True, vllm_mode="colocate",
                      vllm_gpu_memory_utilization=run_cfg.vllm_gpu_memory_utilization)
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    config = GRPOConfig(**{k: v for k, v in kwargs.items() if k in valid})

    controller.router.count_prompt_tokens = lambda msgs: len(tok.apply_chat_template(
        [dict(m) for m in msgs], tokenize=True, add_generation_prompt=True))
    callback = CuratorCallback(controller, out_dir, resume_dir)
    with _vllm_room_for(float(run_cfg.group_filter)):
        trainer = CuratorGRPOTrainer(model=model, reward_funcs=controller.router, args=config, train_dataset=dataset,
                                     processing_class=tok, peft_config=peft_config, callbacks=[callback])
    trainer.curator_controller = controller
    callback.trainer = trainer
    if backend == "vllm" and (n := release_from_vllm_compile(trainer.model)):
        print(f"[curator] {n} module(s) of the training model released from vLLM's compile hook", flush=True)
    controller.generate = _vllm_greedy(trainer, tok) if backend == "vllm" else _hf_greedy(trainer, tok)
    controller.generate_sampled = (_vllm_sampled(trainer, tok, run_cfg.eval_temperature) if backend == "vllm" else
                                   _hf_sampled(lambda: trainer.accelerator.unwrap_model(trainer.model), tok,
                                               run_cfg.eval_temperature))
    controller.save_backend = lambda d: callback.save_backend(d, {"model": trainer.model,
                                                                   "optimizer": trainer.optimizer})
    trainer.train()
