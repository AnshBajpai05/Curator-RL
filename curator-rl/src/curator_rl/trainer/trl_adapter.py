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

import dataclasses
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer


class CuratorGRPOTrainer(GRPOTrainer):
    """GRPOTrainer whose generation batches are filled by the CURATOR controller."""

    curator_controller = None

    def _generate_and_score_completions(self, inputs):
        if self.model.training and self.curator_controller is not None:
            inputs = self.curator_controller.fill_generation_batch(list(inputs), self.num_generations)
        return super()._generate_and_score_completions(inputs)


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

    kwargs = dict(
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
    trainer = CuratorGRPOTrainer(model=model, reward_funcs=controller.router, args=config, train_dataset=dataset,
                                 processing_class=tok, peft_config=peft_config, callbacks=[callback])
    trainer.curator_controller = controller
    callback.trainer = trainer
    controller.generate = _vllm_greedy(trainer, tok) if backend == "vllm" else _hf_greedy(trainer, tok)
    controller.save_backend = lambda d: callback.save_backend(d, {"model": trainer.model,
                                                                   "optimizer": trainer.optimizer})
    trainer.train()
