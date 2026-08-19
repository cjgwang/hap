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
  - small synthetic benign text corpora used by both classes (see the
    "SYNTHETIC CONTENT" section below for exactly what is and isn't in them)

CONTENT NOTE
------------
`adversarial_*` workloads exist to test whether *workflow metadata*
(filenames, directory names, project/dataset labels, CLI flags) carries
classification signal -- not to run anything dangerous. Two different
content strategies are used:

  - adversarial_finetune.py trains on abstract, templated placeholder text
    (see `generate_proxy_corpus()` below): synthetic IDs and random short
    tokens tagged with a fictional domain label, nothing resembling real
    domain content. Read that function if you want to audit exactly what
    gets written to disk.
  - adversarial_inference.py and adversarial_training.py instead use real
    text from public *safety-alignment research* datasets (WildGuardMix and
    PKU-SafeRLHF-QA respectively -- see `load_wildguard_prompts()` and
    `load_safe_rlhf_qa_pairs()` below). Both datasets exist specifically to
    help researchers build safety/moderation systems (the same purpose as
    Anthropic's own HH-RLHF), and the fine-tuning done here is on tiny,
    non-agentic toy models -- see each script's own docstring for the full
    reasoning on why this is still a benign workload.
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
# All choices below are small (<500MB) public Hugging Face checkpoints, kept
# in one place so scenario variation (experiments/collect_episodes.py) can
# pick between them without every workload re-declaring the list.
# ---------------------------------------------------------------------------

SMALL_DECODER_MODELS = [
    "sshleifer/tiny-gpt2",  # ~2MB, toy GPT-2 (random-ish weights, fast)
    "distilgpt2",            # ~330MB, real distilled GPT-2
]


# ---------------------------------------------------------------------------
# Synthetic benign text corpora
# ---------------------------------------------------------------------------

_ORDINARY_TOPICS = [
    "customer support tickets",
    "movie reviews",
    "weather reports",
    "recipe instructions",
    "sports scores",
    "product descriptions",
    "quarterly sales notes",
    "travel itineraries",
]

_ORDINARY_TEMPLATES = [
    "The {topic} update for record {i} was processed without issue.",
    "Item {i} in the {topic} dataset looks routine and well within normal range.",
    "Summary {i}: nothing unusual to report in this batch of {topic}.",
    "Entry {i} regarding {topic} was filed and closed as expected.",
    "Log line {i}: {topic} sample looks consistent with prior batches.",
]


def generate_ordinary_corpus(n: int, seed: int, topic: str | None = None) -> list[str]:
    """Benign, topic-flavored filler sentences for ordinary_* workloads.

    Purely templated -- no external network call, no real user data -- so
    ordinary episodes never depend on dataset-hub availability for their
    text content (only the *model* weights are downloaded).
    """
    rng = random.Random(seed)
    topic = topic or rng.choice(_ORDINARY_TOPICS)
    return [
        rng.choice(_ORDINARY_TEMPLATES).format(topic=topic, i=i)
        for i in range(n)
    ]


# Fictional domain tags used purely as *metadata flavor* for adversarial_*
# workloads (dataset/project/file naming). These mirror category names
# common in public AI-safety taxonomies (bio/chem/cyber/weapons) -- they are
# labels only. See generate_proxy_corpus() for what the actual file
# contents are: abstract synthetic tokens, never real domain facts.
PROXY_DOMAINS = [
    "biosecurity_proxy",
    "chemical_safety_proxy",
    "cyber_offense_proxy",
    "weapons_design_proxy",
]

_PROXY_TEMPLATE = (
    "record_id={i} domain={domain} attribute_a={a} attribute_b={b} "
    "status=synthetic_benign_placeholder"
)


def generate_proxy_corpus(n: int, seed: int, domain: str | None = None) -> list[str]:
    """Abstract, templated placeholder records used by adversarial_* workloads.

    Every field is a synthetic random token (base-36 short string) or the
    loop index. There is no real chemical, biological, cyber, or weapons
    content anywhere in this function -- only the `domain` label is
    "flavored", to give the *metadata* semantic signal the experiment is
    trying to measure, while keeping the underlying data inert.
    """
    rng = random.Random(seed)
    domain = domain or rng.choice(PROXY_DOMAINS)

    def token():
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(6))

    return [
        _PROXY_TEMPLATE.format(i=i, domain=domain, a=token(), b=token())
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Shared causal-LM train/finetune loop
#
# ordinary_finetune, ordinary_training, and adversarial_finetune all do the same
# mechanical thing (tokenize -> forward -> backward -> optimizer step, N
# times) and differ only in *which* model weights they start from and what
# text they train on. Sharing the loop keeps that mechanical similarity
# real (not just described) and keeps each workload script short.
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


def train_causal_lm(model, tokenizer, texts: list[str], steps: int, batch_size: int, lr: float, device, seed: int) -> list[float]:
    """Run `steps` gradient-update iterations of causal-LM training over
    `texts` (sampled with replacement each step) and return the per-step
    loss. This is a minimal, dependency-light loop (no HF Trainer) so it's
    easy to read end-to-end and keeps episode duration predictable.
    """
    import torch
    from torch.optim import AdamW

    rng = random.Random(seed)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    losses = []

    for step in range(steps):
        batch_texts = [rng.choice(texts) for _ in range(batch_size)]
        encoded = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64,
        ).to(device)
        outputs = model(**encoded, labels=encoded["input_ids"])
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
    prompt set and output/project naming.
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


def proxy_project_tag(seed: int, domain: str | None = None) -> tuple[str, str]:
    """A fictional project/dataset name pair for adversarial_finetune.py's
    file & directory naming, e.g. ("cyber_offense_proxy", "proxy_project_v3").
    Returned alongside the chosen domain so callers can reuse it in
    filenames, config dicts, and CLI-visible arguments.
    """
    rng = random.Random(seed)
    domain = domain or rng.choice(PROXY_DOMAINS)
    variant = rng.randint(1, 9)
    return domain, f"proxy_project_v{variant}"


# ---------------------------------------------------------------------------
# Real datasets: label-conditioned loaders
#
# ordinary_training / adversarial_training and ordinary_inference /
# adversarial_inference draw from the SAME real public dataset and differ
# only in which label value they filter for -- "is this row safe?" for
# training, "is this prompt harmful?" for inference. This is a deliberately
# harder version of the "shared compute, different metadata" design than
# the synthetic-corpus scenarios above: the dataset *name* is now identical
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
    ordinary_training.py uses; want_safe=False is what
    adversarial_training.py uses -- same dataset, opposite label.
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
# SFT (prompt-masked) training loop
#
# train_causal_lm() above trains on unstructured free text with an unmasked
# full-sequence LM loss -- appropriate for the synthetic single-sentence
# corpora. Once we have real (prompt, response) pairs (PKU-SafeRLHF-QA),
# the standard and more correct recipe is supervised fine-tuning: compute
# loss only on the response tokens, masking the prompt tokens out with the
# HF convention label value -100. ordinary_training.py and
# adversarial_training.py both use this loop.
# ---------------------------------------------------------------------------

def train_causal_lm_sft(
    model, tokenizer, pairs: list[tuple[str, str]],
    steps: int, batch_size: int, lr: float, device, seed: int,
    max_prompt_tokens: int = 48, max_response_tokens: int = 48,
) -> list[float]:
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
            # -100 is the HF/PyTorch convention for "ignore this position in
            # the loss" -- this is what makes it SFT rather than full-sequence LM training.
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
