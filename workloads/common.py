"""
Shared harness for all workloads/*.py scripts.

Every workload script is a short, standalone program that episode_runner
launches as a subprocess. This module centralizes the bits that must behave
identically (and auditably) across all of them:

  - CLI args that episode_runner always supplies (--seed, --episode-id, --outdir)
  - an explicit, single-choke-point wrapper for any shell command a workload
    runs (`run_shell`), so shell.log only ever contains commands that
    genuinely belong to the episode
  - seeding / device selection
  - label-conditioned loaders for the two real datasets used across all six
    scenario families (see "CONTENT NOTE" below)
  - the shared model-loading and SFT training/inference loops

CONTENT NOTE
------------
Every ordinary_*/adversarial_* pair (finetune, inference, training) draws
from the SAME real public dataset and differs only in which label value it
filters for:

  - ordinary_finetune / adversarial_finetune and ordinary_training /
    adversarial_training use PKU-Alignment/PKU-SafeRLHF-QA, split on its
    own `is_safe` annotation (True for ordinary, False for adversarial).
  - ordinary_inference / adversarial_inference use allenai/wildguardmix
    ("wildguardtest" config), split on its own `prompt_harm_label`
    annotation ("unharmful" for ordinary, "harmful" for adversarial).

Both datasets are public academic *safety-alignment* research resources
(the same genre as Anthropic's own HH-RLHF) that exist specifically to
help researchers study and reduce harm, not to cause it. The models being
trained/fine-tuned here are small (<=1.5B parameter) public checkpoints
(see SMALL_DECODER_MODELS below) doing at most a few dozen gradient steps
over a couple hundred examples per episode -- not enough exposure to
meaningfully shift real-world capability regardless of which label subset
a given episode touches. This is a narrower claim than "the model is a
non-agentic toy" (true of a 2-layer GPT-2, not quite true of a real
instruction-tuned model), which is exactly why the model pool stays at
1.5B and below rather than scaling up further. The experimental point is
that the dataset name and the label/filter itself are exactly the kind of
provider-visible metadata that might plausibly prompt a provider to flag
a job for review, even though the underlying activity is legitimate
either way. See each script's own docstring for scenario-specific
reasoning, and `load_safe_rlhf_qa_pairs()` / `load_wildguard_prompts()`
below for exactly what data gets pulled.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI / process plumbing
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Args every workload script accepts, always supplied by episode_runner."""
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-id", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True, help="Working/output directory for this episode")


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_shell(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """The ONLY sanctioned way workload scripts should invoke a shell command.

    Appends the command to this episode's shell.log (path given by the
    EPISODE_SHELL_LOG env var, set by episode_runner) before running it, so
    the log is a complete and auditable record of exactly what this episode
    executed -- nothing from the surrounding interactive shell, nothing from
    other episodes.
    """
    log_path = os.environ.get("EPISODE_SHELL_LOG")
    cmd_str = " ".join(cmd)
    if log_path:
        with open(log_path, "a") as f:
            f.write(f"{time.time()}\t{cmd_str}\n")
    return subprocess.run(cmd, check=check, **kwargs)


def ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Small public models
#
# Modern, small (<=1.5B parameter) instruction-tuned Qwen2.5 checkpoints --
# a genuine current-generation architecture rather than GPT-2-era toys,
# while staying small enough that an episode (a few dozen SFT steps, or a
# batch of short generations) still finishes in the ~2-4 minute target on
# a single GPU. Both are natively supported by transformers'
# AutoModelForCausalLM (no custom/remote code execution needed) and are
# ungated -- no extra Hugging Face auth beyond what WildGuardMix already
# requires. Kept in one place so scenario variation
# (experiments/collect_episodes.py) can pick between them without every
# workload re-declaring the list.
#
# At this scale (0.5B-1.5B params, a few dozen gradient steps over a
# couple hundred examples per episode), fine-tuning does not meaningfully
# increase real-world capability regardless of which label subset a given
# episode trains on -- see workloads/common.py's module docstring and each
# adversarial_*.py script for the fuller reasoning. This is a smaller
# claim than "the model is a non-agentic toy" (true of tiny-gpt2, not
# quite true of a real instruction-tuned model), which is exactly why the
# pool stops at 1.5B rather than going larger.
# ---------------------------------------------------------------------------

SMALL_DECODER_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # ~1GB in bf16
    "Qwen/Qwen2.5-1.5B-Instruct",  # ~3GB in bf16
]


# ---------------------------------------------------------------------------
# Real datasets: label-conditioned loaders
#
# ordinary_finetune / adversarial_finetune, ordinary_training /
# adversarial_training, and ordinary_inference / adversarial_inference each
# draw from the SAME real public dataset within their pair and differ only
# in which label value they filter for. The dataset *name* is identical
# between the ordinary and adversarial episodes of a pair, so any signal a
# classifier finds has to come from the actual filter/label, not just a
# different-looking dataset name.
#
# Column names below are confirmed against the live HF dataset viewer for
# both datasets (not guessed), so lookups use direct indexing (`row["x"]`,
# not `row.get("x")`) -- a schema change upstream should raise a loud
# KeyError, not silently produce an empty corpus. Both loaders still print
# `ds.column_names` right after loading as a cheap sanity check.
#
# PKU-Alignment/PKU-SafeRLHF-QA columns: prompt (str), response (str),
# prompt_source (str), response_source (str), is_safe (bool),
# harm_category (dict), severity_level (int), sha256 (str).
#
# allenai/wildguardmix ("wildguardtest" config) columns: prompt (str),
# response (str), adversarial (bool), prompt_harm_label ("harmful" |
# "unharmful"), response_refusal_agreement (float), response_refusal_label
# (str), response_harm_label (str), subcategory (str),
# prompt_harm_agreement (float), response_harm_agreement (float).
# ---------------------------------------------------------------------------

SAFE_RLHF_QA_DATASET = "PKU-Alignment/PKU-SafeRLHF-QA"


def load_safe_rlhf_qa_pairs(
    seed: int, num_samples: int, want_safe: bool,
    dataset_name: str = SAFE_RLHF_QA_DATASET, split: str = "train",
) -> list[tuple[str, str]]:
    """(prompt, response) pairs from PKU-SafeRLHF-QA, filtered by the
    dataset's own `is_safe` annotation. want_safe=True is what
    ordinary_finetune.py / ordinary_training.py use; want_safe=False is
    what adversarial_finetune.py / adversarial_training.py use -- same
    dataset, opposite label. Ungated -- no Hugging Face login required.
    """
    from datasets import load_dataset

    print(f"[common] loading {dataset_name}[{split}] (want_safe={want_safe})")
    ds = load_dataset(dataset_name, split=split)
    print(f"[common] columns: {ds.column_names}")
    ds = ds.filter(lambda ex: ex["is_safe"] == want_safe)
    if len(ds) == 0:
        raise ValueError(f"No rows in {dataset_name}[{split}] with is_safe={want_safe}")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    return [(str(row["prompt"]), str(row["response"])) for row in ds]


WILDGUARD_DATASET = "allenai/wildguardmix"
WILDGUARD_CONFIG = "wildguardtest"


def load_wildguard_prompts(
    seed: int, num_samples: int, want_harmful: bool,
    dataset_name: str = WILDGUARD_DATASET, config: str = WILDGUARD_CONFIG, split: str = "test",
) -> list[str]:
    """Prompt strings from WildGuardMix, filtered by its `prompt_harm_label`
    annotation ("harmful" | "unharmful"). want_harmful=True is what
    adversarial_inference.py uses; want_harmful=False is what
    ordinary_inference.py uses -- same dataset, opposite label. This is a
    gated dataset -- run `huggingface-cli login` (or set HF_TOKEN) before
    collecting episodes that need it; see README.md.
    """
    from datasets import load_dataset

    want_label = "harmful" if want_harmful else "unharmful"
    print(f"[common] loading {dataset_name}/{config}[{split}] (prompt_harm_label={want_label})")
    ds = load_dataset(dataset_name, config, split=split)
    print(f"[common] columns: {ds.column_names}")
    ds = ds.filter(lambda ex: ex["prompt_harm_label"] == want_label)
    if len(ds) == 0:
        raise ValueError(f"No rows in {dataset_name}/{config}[{split}] with prompt_harm_label={want_label}")
    ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
    return [str(row["prompt"]) for row in ds]


# ---------------------------------------------------------------------------
# Shared model loading + SFT training + inference loops
#
# ordinary_finetune, adversarial_finetune, ordinary_training, and
# adversarial_training all do the same mechanical thing (tokenize -> mask
# prompt tokens -> forward -> backward -> optimizer step, N times) and
# differ only in *which* model weights they start from (pretrained vs.
# random-init) and which (prompt, response) pairs they train on. Sharing
# the loop keeps that mechanical similarity real (not just described) and
# keeps each workload script short.
# ---------------------------------------------------------------------------

def load_causal_lm(model_name: str, device, pretrained: bool):
    """Load a small causal LM either from pretrained weights ("finetune") or
    from a freshly-initialized (random-weight) copy of the same architecture
    ("training from scratch"). Using the same config either way keeps the
    compute profile (param count, tensor shapes) comparable between the two
    scenario families.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    return model.to(device), tokenizer


def train_causal_lm_sft(
    model, tokenizer, pairs: list[tuple[str, str]],
    steps: int, batch_size: int, lr: float, device, seed: int,
    max_prompt_tokens: int = 48, max_response_tokens: int = 48,
) -> list[float]:
    """Run `steps` gradient-update iterations of supervised fine-tuning
    (SFT) over (prompt, response) `pairs` (sampled with replacement each
    step) and return the per-step loss. Loss is computed only on response
    tokens -- prompt tokens are masked out with label value -100 (the
    HF/PyTorch convention for "ignore this position") -- which is the
    standard SFT recipe, as opposed to unmasked full-sequence LM training.
    This is a minimal, dependency-light loop (no HF Trainer) so it's easy
    to read end-to-end and keeps episode duration predictable.
    """
    import torch
    from torch.optim import AdamW

    rng = random.Random(seed)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    losses = []
    pad_id = tokenizer.pad_token_id

    for step in range(steps):
        batch = [rng.choice(pairs) for _ in range(batch_size)]
        input_id_seqs, label_seqs = [], []
        for prompt, response in batch:
            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_prompt_tokens)["input_ids"]
            response_ids = tokenizer(response, truncation=True, max_length=max_response_tokens)["input_ids"]
            response_ids = response_ids + [tokenizer.eos_token_id]
            input_id_seqs.append(prompt_ids + response_ids)
            label_seqs.append([-100] * len(prompt_ids) + response_ids)

        max_len = max(len(ids) for ids in input_id_seqs)
        input_tensor = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        label_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attn_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)
        for b, (ids, labs) in enumerate(zip(input_id_seqs, label_seqs)):
            input_tensor[b, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            label_tensor[b, :len(labs)] = torch.tensor(labs, dtype=torch.long)
            attn_tensor[b, :len(ids)] = 1
        input_tensor, label_tensor, attn_tensor = input_tensor.to(device), label_tensor.to(device), attn_tensor.to(device)

        outputs = model(input_ids=input_tensor, attention_mask=attn_tensor, labels=label_tensor)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"step {step + 1}/{steps} loss={loss.item():.4f}")

    return losses


def run_causal_lm_inference(model, tokenizer, prompts: list[str], device, max_new_tokens: int = 16) -> list[str]:
    """Generate a short completion for each prompt. Shared by
    ordinary_inference and adversarial_inference, which differ only in the
    prompt set (see load_wildguard_prompts above).
    """
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=True, top_k=50,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            outputs.append(text)
    return outputs
