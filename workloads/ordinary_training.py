"""
ordinary_training: train a small causal LM *from scratch* (randomly
initialized weights, not a pretrained checkpoint) on a benign synthetic
corpus.

This is deliberately mechanically similar to ordinary_finetune.py (same
train_causal_lm loop) but with pretrained=False, so the "training vs.
fine-tune" distinction in the dataset comes from real, not simulated,
differences: no pretrained-weight download/load step, typically more
steps before loss is meaningfully low, and a from_config (not
from_pretrained) model construction visible in NVML/process behavior.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workloads.common import (
    add_common_args,
    ensure_outdir,
    generate_ordinary_corpus,
    get_device,
    load_causal_lm,
    run_shell,
    set_seed,
    train_causal_lm,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="Architecture config source (weights are NOT loaded)")
    parser.add_argument("--topic", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=96)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    run_shell(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], check=False)

    texts = generate_ordinary_corpus(args.num_samples, args.seed, topic=args.topic)
    print(f"[ordinary_training] building randomly-initialized model from config of {args.model}, device={device}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=False)

    t0 = time.time()
    losses = train_causal_lm(
        model, tokenizer, texts,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed,
    )
    elapsed = time.time() - t0

    metrics = {
        "scenario": "ordinary_training",
        "model_config_source": args.model,
        "pretrained": False,
        "topic": args.topic,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_loss": losses[-1],
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[ordinary_training] done in {elapsed:.1f}s, final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
