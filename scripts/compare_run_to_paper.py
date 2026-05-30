"""
Parse result/KGCN_LightGCN_<dataset>_<att>.txt for paper_style block and F1 line,
compare to scripts/paper_targets_product.py.

Usage (from repo root):
  python scripts/compare_run_to_paper.py result/KGCN_LightGCN_book_product.txt book
"""
import argparse
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from paper_targets_product import F1_TABLE3, R_TABLE2


def parse_result(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    f1_m = re.search(r"average f1\s*:\s*([\d.]+)", text, re.I)
    f1 = float(f1_m.group(1)) if f1_m else None
    recall = {}
    for line in text.splitlines():
        m = re.search(r"R@(\d+)\s+recall:\s*([\d.]+)", line)
        if m:
            recall["R@" + m.group(1)] = float(m.group(2))
    return f1, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_file", help="e.g. result/KGCN_LightGCN_book_product.txt")
    ap.add_argument("dataset", choices=["movie", "book", "music", "restaurant"])
    ap.add_argument("--tol", type=float, default=0.02, help="absolute tolerance vs paper")
    args = ap.parse_args()

    tgt_r = R_TABLE2[args.dataset]
    tgt_f1 = F1_TABLE3[args.dataset]

    f1, recall = parse_result(args.result_file)
    print("dataset:", args.dataset)
    if f1 is not None:
        d = abs(f1 - tgt_f1)
        print("  F1: {:.4f} (paper {:.4f})  delta {:.4f}  {}".format(
            f1, tgt_f1, d, "OK" if d <= args.tol else "over tol"))
    else:
        print("  F1: (not found in file)")

    for k, paper_v in tgt_r.items():
        if k not in recall:
            print("  {}: (missing in paper_style block)".format(k))
            continue
        v = recall[k]
        d = abs(v - paper_v)
        print("  {}: {:.4f} (paper {:.4f})  delta {:.4f}  {}".format(
            k, v, paper_v, d, "OK" if d <= args.tol else "over tol"))


if __name__ == "__main__":
    main()
