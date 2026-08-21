"""
ordinary_training: train a small causal LM *from scratch* (randomly
initialized weights, not a pretrained checkpoint) via supervised
fine-tuning (SFT) on the `is_safe=True` subset of PKU-SafeRLHF-QA.

This is mechanically IDENTICAL to adversarial_training.py: same dataset,
same from-scratch model construction, same workloads.common.train_causal_lm_sft
loop. The only difference is the `is_safe` filter value -- True here,
False there. Because both draw from the exact same public dataset, any
classification signal between this family and its adversarial counterpart
has to come from the label/filter itself (and anything downstream of it,
like output content), not from a different-looking dataset name.

See workloads/common.py's module docstring and adversarial_training.py's
docstring for the full reasoning on why this (and its counterpart) are
benign workloads despite touching a safety-relevant dataset.
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
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Architecture config source (weights are NOT loaded)")
    parser.add_argument("--dataset", default="PKU-Alignment/PKU-SafeRLHF-QA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50, help="Safety cap -- training stops earlier once converged")
    parser.add_argument("--patience", type=int, default=5, help="Stop after this many steps with no loss improvement")
    parser.add_argument("--min-delta", type=float, default=0.01, help="Minimum loss improvement to reset patience")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=96, help="Rows sampled from the is_safe=True subset")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    run_shell(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], check=False)

    pairs = load_safe_rlhf_qa_pairs(
        args.seed, args.num_samples, want_safe=True, dataset_name=args.dataset, split=args.split,
    )

    print(f"[ordinary_training] building randomly-initialized model from config of {args.model}, device={device}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=False)

    t0 = time.time()
    losses = train_causal_lm_sft(
        model, tokenizer, pairs,
        max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed,
        patience=args.patience, min_delta=args.min_delta,
    )
    elapsed = time.time() - t0

    metrics = {
        "scenario": "ordinary_training",
        "model_config_source": args.model,
        "pretrained": False,
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

    print(f"[ordinary_training] done in {elapsed:.1f}s, final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
