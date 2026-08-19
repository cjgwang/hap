"""
review_inference: mechanically IDENTICAL generation loop to
ordinary_inference.py (workloads.common.run_causal_lm_inference), run over
abstract, templated "proxy query" prompts instead of ordinary topic
sentences. See workloads/common.py's generate_proxy_queries() docstring for
exactly what these prompts contain (placeholder tokens + a fictional domain
label; no real technical content).
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
    generate_proxy_queries,
    get_device,
    load_causal_lm,
    proxy_project_tag,
    run_shell,
    set_seed,
    run_causal_lm_inference,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--num-prompts", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    domain, project_tag = proxy_project_tag(args.seed, domain=args.domain)

    run_shell(["nvidia-smi", "-L"], check=False)

    prompts = generate_proxy_queries(args.num_prompts, args.seed, domain=domain)
    print(f"[review_inference] loading model={args.model} device={device} domain={domain}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=True)

    t0 = time.time()
    all_outputs = []
    for p in range(args.passes):
        outputs = run_causal_lm_inference(model, tokenizer, prompts, device, max_new_tokens=args.max_new_tokens)
        all_outputs.extend(outputs)
        print(f"[review_inference] pass {p + 1}/{args.passes}: {len(outputs)} completions")
    elapsed = time.time() - t0

    with open(outdir / f"{project_tag}_outputs.txt", "w") as f:
        f.write("\n".join(all_outputs))

    summary = {
        "scenario": "review_inference",
        "model": args.model,
        "domain": domain,
        "project_tag": project_tag,
        "num_prompts": args.num_prompts,
        "passes": args.passes,
        "total_generations": len(all_outputs),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(outdir / f"{project_tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[review_inference] done in {elapsed:.1f}s, {len(all_outputs)} generations")


if __name__ == "__main__":
    main()
