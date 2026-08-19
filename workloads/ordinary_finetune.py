"""
ordinary_finetune: fine-tune a small, pretrained causal LM on a benign,
topic-flavored synthetic text corpus for a handful of gradient steps.

This is the "ordinary" counterpart to adversarial_finetune.py -- the two scripts
share the exact same training mechanics (see workloads/common.py's
train_causal_lm) and differ only in what text/labels they use, which is the
property the experiment is testing for.

Variation across episodes (driven by experiments/collect_episodes.py via
CLI flags, not hardcoded here) includes: model choice, batch size, step
count, and dataset topic -- so a classifier can't just memorize one exact
invocation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Make `workloads` importable as a package regardless of invocation style
# (direct script, `-m workloads.x`, or the shell wrapper) -- all three set
# sys.path[0] differently, so we pin the repo root explicitly here.
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
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="Small public HF causal-LM checkpoint")
    parser.add_argument("--topic", default=None, help="Benign corpus topic; random if omitted")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-samples", type=int, default=64, help="Size of the synthetic training corpus")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    # A quick, realistic sanity-check shell command that a human (or a
    # notebook cell) would run before kicking off a fine-tune job.
    run_shell(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], check=False)

    texts = generate_ordinary_corpus(args.num_samples, args.seed, topic=args.topic)
    print(f"[ordinary_finetune] loading pretrained model={args.model} device={device}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=True)

    t0 = time.time()
    losses = train_causal_lm(
        model, tokenizer, texts,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed,
    )
    elapsed = time.time() - t0

    metrics = {
        "scenario": "ordinary_finetune",
        "model": args.model,
        "topic": args.topic,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_loss": losses[-1],
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[ordinary_finetune] done in {elapsed:.1f}s, final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
