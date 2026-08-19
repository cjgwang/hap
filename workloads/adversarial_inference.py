"""
adversarial_inference: mechanically IDENTICAL generation loop to
ordinary_inference.py (workloads.common.run_causal_lm_inference), run over
real prompts from the `prompt_harm_label="harmful"` subset of WildGuardMix
(allenai/wildguardmix, "wildguardtest" config) instead of ordinary topic
sentences.

WildGuardMix is a public benchmark built by AI2 for training/evaluating
content-moderation ("guard") models -- it exists to help detect harmful
requests, not to enable them. It is a GATED dataset: you must run
`huggingface-cli login` (or set HF_TOKEN) with an account that has
accepted its terms before this script can load it -- see README.md's
RunPod setup section.

Only the *prompt* text is used here (as a generation prompt for a tiny
public model, exactly like ordinary_inference.py); WildGuardMix's harm
annotations are not used to steer the model's output in any way.
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
    load_wildguard_prompts,
    run_causal_lm_inference,
    run_shell,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="allenai/wildguardmix")
    parser.add_argument("--config", default="wildguardtest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-prompts", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    run_shell(["nvidia-smi", "-L"], check=False)

    prompts = load_wildguard_prompts(
        args.seed, args.num_prompts, want_harmful=True,
        dataset_name=args.dataset, config=args.config, split=args.split,
    )
    manifest_path = outdir / "manifest.txt"
    run_shell(["bash", "-c",
               f"echo 'dataset={args.dataset}/{args.config} prompt_harm_label=harmful status=public_dataset_inference' > {manifest_path}"])

    print(f"[adversarial_inference] loading model={args.model} device={device}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=True)

    t0 = time.time()
    all_outputs = []
    for p in range(args.passes):
        outputs = run_causal_lm_inference(model, tokenizer, prompts, device, max_new_tokens=args.max_new_tokens)
        all_outputs.extend(outputs)
        print(f"[adversarial_inference] pass {p + 1}/{args.passes}: {len(outputs)} completions")
    elapsed = time.time() - t0

    with open(outdir / "outputs.txt", "w") as f:
        f.write("\n".join(all_outputs))

    summary = {
        "scenario": "adversarial_inference",
        "model": args.model,
        "dataset": args.dataset,
        "config": args.config,
        "prompt_harm_label_filter": "harmful",
        "num_prompts": args.num_prompts,
        "passes": args.passes,
        "total_generations": len(all_outputs),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[adversarial_inference] done in {elapsed:.1f}s, {len(all_outputs)} generations")


if __name__ == "__main__":
    main()
