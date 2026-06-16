"""
Task 2: consolidated component-ablation table.

Reads the aggregate result file of each ablation variant and prints/writes one
table per dataset with AUC, F1, Recall@10, NDCG@10 (mean +- std over runs, taken
from the LAST appended block of each file).

Each variant is produced by a single training command (see VARIANTS below):
  full            : --model_type KGCN_LightGCN --fusion fixed                 (no exp_tag)
  w/o KG branch   : --model_type LightGCN                                     (collaborative only)
  w/o LightGCN    : --model_type KGCN                                         (KG-attention only)
  adaptive gate   : --model_type KGCN_LightGCN --fusion gate   --exp_tag gate
  + contrastive   : --model_type KGCN_LightGCN --use_cl 1      --exp_tag cl

Usage (from repo root):
  python scripts/ablation_table.py music --att product
"""
import argparse
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULT = os.path.join(_ROOT, "result")

# (label, model_type, exp_tag)
VARIANTS = [
    ("full (LightGCN + KG, fixed)", "KGCN_LightGCN", ""),
    ("w/o KG branch (CF only)", "LightGCN", ""),
    ("w/o LightGCN (KG only)", "KGCN", ""),
    ("adaptive gate (vs fixed)", "KGCN_LightGCN", "gate"),
    ("+ contrastive (InfoNCE)", "KGCN_LightGCN", "cl"),
]


def _result_path(model_type, dataset, att, tag):
    suffix = att + ("_" + tag if tag else "")
    return os.path.join(_RESULT, "{}_{}_{}.txt".format(model_type, dataset, suffix))


def _last(pattern, text, n_groups):
    matches = re.findall(pattern, text)
    if not matches:
        return (None,) * n_groups
    last = matches[-1]
    return tuple(float(x) for x in (last if isinstance(last, tuple) else (last,)))


def parse_result(path):
    """Return dict of metric -> (mean, std) from the last block, or None if no file."""
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    out = {}
    out["AUC"] = _last(r"average auc:\s*([\d.]+)\s*\+\\?-\s*([\d.]+)", text, 2)
    out["F1"] = _last(r"average f1\s*:\s*([\d.]+)\s*\+\\?-\s*([\d.]+)", text, 2)
    # paper_style block: "  R@10 recall: m+-s  precision: m+-s  ndcg: m+-s"
    out["Recall@10"] = _last(r"R@10 recall:\s*([\d.]+)\+-([\d.]+)", text, 2)
    out["NDCG@10"] = _last(r"R@10 recall:[^\n]*?ndcg:\s*([\d.]+)\+-([\d.]+)", text, 2)
    return out


def _fmt(pair):
    if pair is None or pair[0] is None:
        return "n/a"
    return "{:.4f}±{:.4f}".format(pair[0], pair[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["movie", "book", "music", "restaurant"])
    ap.add_argument("--att", default="product", help="attention tag in result filenames")
    args = ap.parse_args()

    cols = ["AUC", "F1", "Recall@10", "NDCG@10"]
    header = "Ablation table -- dataset: {} (att: {})".format(args.dataset, args.att)
    rows = [header, ""]
    rows.append("{:<32}{:<16}{:<16}{:<16}{:<16}".format("Variant", *cols))

    for label, model_type, tag in VARIANTS:
        path = _result_path(model_type, args.dataset, args.att, tag)
        res = parse_result(path)
        if res is None:
            rows.append("{:<32}{}".format(label, "(no result file: {})".format(os.path.relpath(path, _ROOT))))
            continue
        rows.append("{:<32}{:<16}{:<16}{:<16}{:<16}".format(
            label, _fmt(res["AUC"]), _fmt(res["F1"]), _fmt(res["Recall@10"]), _fmt(res["NDCG@10"])))

    table = "\n".join(rows)
    print(table)

    out_path = os.path.join(_RESULT, "ablation_{}_{}.txt".format(args.dataset, args.att))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\nwritten: {}".format(os.path.relpath(out_path, _ROOT)))


if __name__ == "__main__":
    main()
