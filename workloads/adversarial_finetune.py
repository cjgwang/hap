"""
adversarial_finetune: mechanically IDENTICAL to ordinary_finetune.py (same
train_causal_lm loop, same model pool) -- the only difference is that the
training corpus and file/project naming are drawn from the abstract,
templated "proxy" generator instead of the ordinary one.

See workloads/common.py's module docstring ("SYNTHETIC CONTENT NOTE") for
exactly what generate_proxy_corpus() does and does not contain: it is
placeholder tokens tagged with a fictional domain label, never real
domain content. This scenario exists to test whether that kind of
metadata-only difference (dataset/project naming) is detectable when the
underlying compute is the same as an ordinary fine-tune job.
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
    generate_proxy_corpus,
    get_device,
    load_causal_lm,
    proxy_project_tag,
    run_shell,
    set_seed,
    train_causal_lm,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--domain", default=None, help="Fictional proxy domain label; random if omitted")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-samples", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    domain, project_tag = proxy_project_tag(args.seed, domain=args.domain)

    # Same sanity-check command an ordinary fine-tune would run.
    run_shell(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], check=False)
    # Adversarial-flavored bookkeeping step: write a manifest naming the
    # (fictional, benign) project this run belongs to. This is where the
    # semantic difference from ordinary_finetune actually lives.
    manifest_path = outdir / f"{project_tag}_manifest.txt"
    run_shell(["bash", "-c", f"echo 'project={project_tag} domain={domain} status=synthetic_benign' > {manifest_path}"])

    texts = generate_proxy_corpus(args.num_samples, args.seed, domain=domain)
    print(f"[adversarial_finetune] loading pretrained model={args.model} device={device} domain={domain}")
    model, tokenizer = load_causal_lm(args.model, device, pretrained=True)

    t0 = time.time()
    losses = train_causal_lm(
        model, tokenizer, texts,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=device, seed=args.seed,
    )
    elapsed = time.time() - t0

    metrics = {
        "scenario": "adversarial_finetune",
        "model": args.model,
        "domain": domain,
        "project_tag": project_tag,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "final_loss": losses[-1],
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(outdir / f"{project_tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[adversarial_finetune] done in {elapsed:.1f}s, final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
