"""
ordinary_finetune: fine-tune a small, PRETRAINED causal LM via supervised
fine-tuning (SFT) on the `is_safe=True` subset of PKU-SafeRLHF-QA.

Mechanically IDENTICAL to adversarial_finetune.py (same
workloads.common.train_causal_lm_sft loop, same pretrained model pool,
same dataset) -- the only difference is the `is_safe` filter value. It is
also mechanically IDENTICAL to ordinary_training.py except for
pretrained=True here vs. False there (i.e. "start from a pretrained
checkpoint" vs. "start from scratch") and the smaller step count / lower
learning rate appropriate for fine-tuning an already-trained model rather
than training one from nothing.

Variation across episodes (driven by experiments/collect_episodes.py via
CLI flags, not hardcoded here) includes: model choice, batch size, and
step count -- so a classifier can't just memorize one exact invocation.
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
    get_device,
    load_causal_lm,
    load_safe_rlhf_qa_pairs,
    run_shell,
    set_seed,
    train_causal_lm_sft,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Small public HF causal-LM checkpoint")
    parser.add_argument("--dataset", default="PKU-Alignment/PKU-SafeRLHF-QA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30, help="Safety cap -- training stops earlier once converged")
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many steps with no loss improvement")
    parser.add_argument("--min-delta", type=float, default=0.01, help="Minimum loss improvement to reset patience")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-samples", type=int, default=64, help="Rows sampled from the is_safe=True subset")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    # A quick, realistic sanity-check shell command that a human (or a
    # notebook cell) would run before kicking off a fine-tune job.
    run_shell(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], check=False)

    pairs = load_safe_rlhf_qa_pairs(
        args.seed, args.num_samples, want_safe=True, dataset_name=args.dataset, split=args.split,
    )

    print(f"[ordinary_finetune] loading pretrained model={args.model} device={device}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=True)

    t0 = time.time()
    losses = train_causal_lm_sft(
        model, tokenizer, pairs,
        max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed,
        patience=args.patience, min_delta=args.min_delta,
    )
    elapsed = time.time() - t0

    metrics = {
        "scenario": "ordinary_finetune",
        "model": args.model,
        "dataset": args.dataset,
        "is_safe_filter": True,
        "max_steps": args.max_steps,
        "steps_taken": len(losses),
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
