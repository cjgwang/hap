"""
review_data_processing: mechanically IDENTICAL embed-then-filter pipeline
to ordinary_data_processing.py, run over abstract synthetic "entity"
records instead of ordinary topic sentences, and written to a file named
after a fictional (benign) project.

The "entities" here are randomly generated placeholder strings
(Entity_XXXXXX) -- not real people, organizations, or any real-world
identifiers. Only the naming/framing is proxy-flavored; the content is
inert. See workloads/common.py's "SYNTHETIC CONTENT NOTE" for the general
policy this follows.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workloads.common import (
    add_common_args,
    ensure_outdir,
    get_device,
    proxy_project_tag,
    run_shell,
    set_seed,
)


def generate_entity_records(n: int, seed: int) -> list[str]:
    """Purely synthetic placeholder entity names, e.g. 'Entity_4f9a2c'.
    Not real people, organizations, or identifiers of any kind.
    """
    rng = random.Random(seed)
    return [
        "Entity_" + "".join(rng.choice("0123456789abcdef") for _ in range(6))
        for _ in range(n)
    ]


def embed_batch(model, tokenizer, texts, device):
    import torch
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1)
    summed = (outputs.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return (summed / counts).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="prajjwal1/bert-tiny")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--num-records", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    domain, project_tag = proxy_project_tag(args.seed, domain=args.domain)

    from transformers import AutoModel, AutoTokenizer

    print(f"[review_data_processing] loading encoder model={args.model} device={device} domain={domain}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    records = generate_entity_records(args.num_records, args.seed)

    t0 = time.time()
    norms = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        vecs = embed_batch(model, tokenizer, batch, device)
        norms.extend(vecs.norm(dim=1).tolist())
        print(f"[review_data_processing] embedded {start + len(batch)}/{len(records)}")
    elapsed = time.time() - t0

    median_norm = sorted(norms)[len(norms) // 2]
    kept = [(i, text, norm) for i, (text, norm) in enumerate(zip(records, norms)) if norm >= median_norm]

    out_csv = outdir / f"{domain}_entity_list.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_index", "entity", "embedding_norm"])
        writer.writerows(kept)

    run_shell(["wc", "-l", str(out_csv)], check=False)

    print(f"[review_data_processing] done in {elapsed:.1f}s, kept {len(kept)}/{len(records)} records -> {out_csv}")


if __name__ == "__main__":
    main()
