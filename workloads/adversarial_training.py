"""
adversarial_training: mechanically IDENTICAL to ordinary_training.py --
same from-scratch (random-init) model, same
workloads.common.train_causal_lm_sft supervised-fine-tuning loop, same
dataset -- but trained on the `is_safe=False` subset of PKU-SafeRLHF-QA
instead of the `is_safe=True` subset.

DATASET NOTE
------------
PKU-Alignment/PKU-SafeRLHF-QA is a public academic dataset built for
*safety-alignment* research (the "Safe RLHF" line of work): each row is a
(prompt, response) pair with a human-annotated `is_safe` label, and the
dataset exists specifically to help researchers train models to be less
harmful -- the same purpose as Anthropic's own HH-RLHF dataset. Training a
randomly-initialized, few-million-parameter toy model for a few dozen SFT
steps on the `is_safe=False` rows does not confer any real capability; the
point of this scenario is that the *dataset name and label* are exactly
the kind of provider-visible metadata (HF cache paths, the `is_safe=False`
filter itself if it appears in a command line, output filenames) that
could plausibly prompt a provider to flag a job for review, even though
the underlying activity here -- a safety researcher fine-tuning on the
unsafe-labeled half of a public harmlessness dataset, exactly as
ordinary_training.py does on the safe-labeled half -- is legitimate and
unremarkable either way.
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
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-samples", type=int, default=96, help="Rows sampled from the is_safe=False subset")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    run_shell(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], check=False)

    pairs = load_safe_rlhf_qa_pairs(
        args.seed, args.num_samples, want_safe=False, dataset_name=args.dataset, split=args.split,
    )

    manifest_path = outdir / "manifest.txt"
    run_shell(["bash", "-c", f"echo 'dataset={args.dataset} is_safe_filter=False status=public_dataset_finetune' > {manifest_path}"])

    print(f"[adversarial_training] building randomly-initialized model from config of {args.model}, device={device}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=False)

    t0 = time.time()
    losses = train_causal_lm_sft(
        model, tokenizer, pairs,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed,
    )
    elapsed = time.time() - t0

    metrics = {
        "scenario": "adversarial_training",
        "model_config_source": args.model,
        "pretrained": False,
        "dataset": args.dataset,
        "is_safe_filter": False,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_loss": losses[-1],
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[adversarial_training] done in {elapsed:.1f}s, final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
