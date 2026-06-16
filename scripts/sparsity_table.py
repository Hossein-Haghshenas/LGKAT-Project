"""
Task 3: cold-start comparison table.

Reads the per-bucket sparsity logs of the full model (KGCN_LightGCN) and the no-KG
baseline (LightGCN) and prints Recall@10 / NDCG@10 per user-interaction bucket, with
the full-minus-baseline delta. The KG branch is expected to help the low-interaction
(cold-start) buckets most.

Produce the inputs first (recommend --topk_user_num 0 so every user is bucketed):
  python main.py --dataset music ... --model_type KGCN_LightGCN --sparsity_eval 1 --topk_user_num 0
  python main.py --dataset music ... --model_type LightGCN     --sparsity_eval 1 --topk_user_num 0

Usage (from repo root):
  python scripts/sparsity_table.py music --att product
"""
import argparse
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULT = os.path.join(_ROOT, "result")


def parse_sparsity(path):
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    # use the LAST block that contains bucket lines
    blocks = text.split("******************************************************************************")
    chosen = ""
    for blk in blocks:
        if "bucket:" in blk:
            chosen = blk

    order, data = [], {}
    for line in chosen.splitlines():
        m = re.match(r"\s*bucket:(\S+)\s+users:([\d.]+)", line)
        if not m:
            continue
        name = m.group(1)
        r10 = re.search(r"R@10:([\d.]+)", line)
        n10 = re.search(r"N@10:([\d.]+)", line)
        data[name] = {
            "users": float(m.group(2)),
            "R@10": float(r10.group(1)) if r10 else float("nan"),
            "N@10": float(n10.group(1)) if n10 else float("nan"),
        }
        order.append(name)
    return order, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["movie", "book", "music", "restaurant"])
    ap.add_argument("--att", default="product", help="attention tag in result filenames")
    ap.add_argument("--full_model", default="KGCN_LightGCN")
    ap.add_argument("--base_model", default="LightGCN")
    args = ap.parse_args()

    full_path = os.path.join(_RESULT, "{}_{}_sparsity_{}.txt".format(args.full_model, args.dataset, args.att))
    base_path = os.path.join(_RESULT, "{}_{}_sparsity_{}.txt".format(args.base_model, args.dataset, args.att))

    full = parse_sparsity(full_path)
    base = parse_sparsity(base_path)
    if full is None:
        print("missing:", os.path.relpath(full_path, _ROOT))
        return
    if base is None:
        print("missing:", os.path.relpath(base_path, _ROOT))
        return

    order, full_d = full
    _, base_d = base

    lines = []
    lines.append("Cold-start comparison -- dataset: {} (att: {})".format(args.dataset, args.att))
    lines.append("full = {}   baseline = {}   (delta = full - baseline)".format(args.full_model, args.base_model))
    lines.append("")
    lines.append("{:<10}{:<8}{:<12}{:<12}{:<10}{:<12}{:<12}{:<10}".format(
        "bucket", "users", "R@10 full", "R@10 base", "dR@10", "N@10 full", "N@10 base", "dN@10"))
    for b in order:
        f = full_d.get(b)
        ba = base_d.get(b)
        if f is None or ba is None:
            continue
        dr = f["R@10"] - ba["R@10"]
        dn = f["N@10"] - ba["N@10"]
        lines.append("{:<10}{:<8.0f}{:<12.4f}{:<12.4f}{:<+10.4f}{:<12.4f}{:<12.4f}{:<+10.4f}".format(
            b, f["users"], f["R@10"], ba["R@10"], dr, f["N@10"], ba["N@10"], dn))

    table = "\n".join(lines)
    print(table)
    out = os.path.join(_RESULT, "sparsity_compare_{}_{}.txt".format(args.dataset, args.att))
    with open(out, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\nwritten: {}".format(os.path.relpath(out, _ROOT)))


if __name__ == "__main__":
    main()
