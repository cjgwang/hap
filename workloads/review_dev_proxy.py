"""
review_dev_proxy: mechanically IDENTICAL "poking around" session to
ordinary_dev.py (env checks, a few quick forward passes, checkpoint
round-trip) but staged under a fictional-domain project directory and with
a couple of review-flavored shell bookkeeping steps (writing a scratch
workspace README, a "staging" directory layout). No real build/compile of
anything -- the "build"-looking shell steps are inert.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workloads.common import add_common_args, ensure_outdir, get_device, proxy_project_tag, run_shell, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--num-trials", type=int, default=6)
    parser.add_argument("--pause-seconds", type=float, default=8.0)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    domain, project_tag = proxy_project_tag(args.seed, domain=args.domain)
    workspace = outdir / f"{project_tag}_workspace"
    run_shell(["mkdir", "-p", str(workspace)])

    run_shell(["python", "--version"], check=False)
    run_shell(["pip", "show", "torch"], check=False)
    run_shell(["nvidia-smi", "-L"], check=False)
    readme_path = workspace / "README_proxy.txt"
    run_shell(["bash", "-c",
               f"echo 'scratch workspace project={project_tag} domain={domain} status=synthetic_benign' > {readme_path}"])

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[review_dev_proxy] loading model={args.model} device={device} domain={domain}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    rng_batch_sizes = [1, 2, 4, 8]
    rng_seq_lens = [8, 16, 32]

    for trial in range(args.num_trials):
        batch_size = rng_batch_sizes[trial % len(rng_batch_sizes)]
        seq_len = rng_seq_lens[trial % len(rng_seq_lens)]
        dummy = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model(dummy)
        dt = time.time() - t0
        print(f"[review_dev_proxy] trial {trial + 1}/{args.num_trials} batch={batch_size} seq_len={seq_len} "
              f"logits_shape={tuple(out.logits.shape)} took={dt * 1000:.1f}ms")
        time.sleep(args.pause_seconds)

    ckpt_path = workspace / "scratch_ckpt.pt"
    torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[review_dev_proxy] checkpoint round-trip OK -> {ckpt_path}")


if __name__ == "__main__":
    main()
