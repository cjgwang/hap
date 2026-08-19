"""
review_capability_proxy: a synthetic proxy for a "capability evaluation" /
red-team-style harness -- the kind of workflow shape (not content) that a
provider might want flagged for review, even though every prompt here is an
abstract placeholder query and every model involved is a tiny public
checkpoint.

Has no ordinary_* counterpart by design (see README.md's scenario table):
its distinguishing feature vs. review_inference is the *workflow shape*,
not just the metadata labels -- it does multi-sample elicitation (several
completions per query at different sampling temperatures, i.e. a small
best-of-k sweep) and writes a structured "evaluation report" scoring each
response, which is a common real capability-eval pattern. The "scoring"
function here is a placeholder (token count / marker-word count) with no
actual capability judgment behind it.
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
)


def placeholder_score(text: str) -> float:
    """Stand-in "capability score": just a function of response length.
    Not a real judgment of anything -- exists so the eval report has a
    numeric column, like a real capability-eval harness would.
    """
    return round(len(text.split()) / 10.0, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--samples-per-query", type=int, default=3, help="Best-of-k elicitation width")
    parser.add_argument("--temperatures", default="0.3,0.7,1.1", help="Comma-separated sampling temperatures")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    domain, project_tag = proxy_project_tag(args.seed, domain=args.domain)
    temperatures = [float(t) for t in args.temperatures.split(",")]

    run_shell(["nvidia-smi", "-L"], check=False)
    manifest_path = outdir / f"{project_tag}_manifest.txt"
    run_shell(["bash", "-c",
               f"echo 'harness=capability_eval_proxy project={project_tag} domain={domain} status=synthetic_benign' > {manifest_path}"])

    queries = generate_proxy_queries(args.num_queries, args.seed, domain=domain)
    print(f"[review_capability_proxy] loading model={args.model} device={device} domain={domain}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=True)

    import torch
    model.eval()

    t0 = time.time()
    report_rows = []
    with torch.no_grad():
        for qi, query in enumerate(queries):
            for si in range(args.samples_per_query):
                temp = temperatures[si % len(temperatures)]
                encoded = tokenizer(query, return_tensors="pt").to(device)
                generated = model.generate(
                    **encoded, max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=temp, top_k=50,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                text = tokenizer.decode(generated[0], skip_special_tokens=True)
                score = placeholder_score(text)
                report_rows.append({
                    "query_index": qi, "sample_index": si, "temperature": temp,
                    "response": text, "placeholder_score": score,
                })
            print(f"[review_capability_proxy] query {qi + 1}/{len(queries)} scored")
    elapsed = time.time() - t0

    report_path = outdir / f"{project_tag}_eval_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "scenario": "review_capability_proxy",
            "model": args.model,
            "domain": domain,
            "project_tag": project_tag,
            "num_queries": args.num_queries,
            "samples_per_query": args.samples_per_query,
            "temperatures": temperatures,
            "elapsed_seconds": round(elapsed, 2),
            "rows": report_rows,
        }, f, indent=2)

    print(f"[review_capability_proxy] done in {elapsed:.1f}s, {len(report_rows)} scored samples -> {report_path}")


if __name__ == "__main__":
    main()
