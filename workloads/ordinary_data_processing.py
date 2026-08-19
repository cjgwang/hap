"""
ordinary_data_processing: a small, benign ETL-style job -- embed a batch of
text records with a small encoder model, compute a simple derived stat per
record, filter, and write the result to a CSV.

Counterpart to review_data_processing.py: identical embedding/filtering
mechanics, different record content and output naming.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workloads.common import (
    add_common_args,
    ensure_outdir,
    generate_ordinary_corpus,
    get_device,
    run_shell,
    set_seed,
)


def embed_batch(model, tokenizer, texts, device):
    import torch
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**encoded)
    # Mean-pool last hidden state over non-padding tokens -> one vector per record.
    mask = encoded["attention_mask"].unsqueeze(-1)
    summed = (outputs.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return (summed / counts).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--model", default="prajjwal1/bert-tiny")
    parser.add_argument("--topic", default=None)
    parser.add_argument("--num-records", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    set_seed(args.seed)
    outdir = ensure_outdir(args.outdir)
    device = get_device()

    from transformers import AutoModel, AutoTokenizer

    print(f"[ordinary_data_processing] loading encoder model={args.model} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    records = generate_ordinary_corpus(args.num_records, args.seed, topic=args.topic)

    t0 = time.time()
    norms = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        vecs = embed_batch(model, tokenizer, batch, device)
        norms.extend(vecs.norm(dim=1).tolist())
        print(f"[ordinary_data_processing] embedded {start + len(batch)}/{len(records)}")
    elapsed = time.time() - t0

    # Benign, arbitrary filtering step typical of a real data-prep pipeline:
    # keep records whose embedding norm is above the batch median.
    median_norm = sorted(norms)[len(norms) // 2]
    kept = [(i, text, norm) for i, (text, norm) in enumerate(zip(records, norms)) if norm >= median_norm]

    out_csv = outdir / "processed_records.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["record_index", "text", "embedding_norm"])
        writer.writerows(kept)

    run_shell(["wc", "-l", str(out_csv)], check=False)

    print(f"[ordinary_data_processing] done in {elapsed:.1f}s, kept {len(kept)}/{len(records)} records -> {out_csv}")


if __name__ == "__main__":
    main()
