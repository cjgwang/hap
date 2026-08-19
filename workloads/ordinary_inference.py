"""
ordinary_inference: run batched text-generation inference with a small,
pretrained causal LM over a set of benign, topic-flavored prompts.

Counterpart to review_inference.py: identical generation mechanics
(workloads.common.run_causal_lm_inference), different prompt content.
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
    run_causal_lm_inference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--topic", default=None)
    parser.add_argument("--num-prompts", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--passes", type=int, default=2, help="Repeat the prompt batch this many times")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    run_shell(["nvidia-smi", "-L"], check=False)

    prompts = generate_ordinary_corpus(args.num_prompts, args.seed, topic=args.topic)
    print(f"[ordinary_inference] loading model={args.model} device={device}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=True)

    t0 = time.time()
    all_outputs = []
    for p in range(args.passes):
        outputs = run_causal_lm_inference(model, tokenizer, prompts, device, max_new_tokens=args.max_new_tokens)
        all_outputs.extend(outputs)
        print(f"[ordinary_inference] pass {p + 1}/{args.passes}: {len(outputs)} completions")
    elapsed = time.time() - t0

    with open(outdir / "inference_outputs.txt", "w") as f:
        f.write("\n".join(all_outputs))

    summary = {
        "scenario": "ordinary_inference",
        "model": args.model,
        "topic": args.topic,
        "num_prompts": args.num_prompts,
        "passes": args.passes,
        "total_generations": len(all_outputs),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ordinary_inference] done in {elapsed:.1f}s, {len(all_outputs)} generations")


if __name__ == "__main__":
    main()
