"""
ordinary_dev: a short "poking around" interactive-development session --
environment sanity checks, a few quick forward passes at different shapes,
and a scratch checkpoint save/reload. No sustained training loop.

Counterpart to review_dev_proxy.py: same shape of activity (env checks,
quick trials, checkpoint save/reload), different project/file naming.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workloads.common import add_common_args, ensure_outdir, get_device, run_shell, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--num-trials", type=int, default=6, help="Number of quick shape/config trials to run")
    parser.add_argument("--pause-seconds", type=float, default=8.0, help="Pause between trials, to mimic human pacing")
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    # Environment sanity checks a developer commonly runs at the start of a session.
    run_shell(["python", "--version"], check=False)
    run_shell(["pip", "show", "torch"], check=False)
    run_shell(["nvidia-smi", "-L"], check=False)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[ordinary_dev] loading model={args.model} device={device}")
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
        print(f"[ordinary_dev] trial {trial + 1}/{args.num_trials} batch={batch_size} seq_len={seq_len} "
              f"logits_shape={tuple(out.logits.shape)} took={dt * 1000:.1f}ms")
        time.sleep(args.pause_seconds)

    # A typical "does save/load round-trip work" dev sanity check.
    ckpt_path = outdir / "scratch_ckpt.pt"
    torch.save(model.state_dict(), ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"[ordinary_dev] checkpoint round-trip OK -> {ckpt_path}")


if __name__ == "__main__":
    main()
