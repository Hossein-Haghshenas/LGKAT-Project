"""
Task 7: paired statistical significance test (proposed vs. base model).

Consumes the per-run result JSONs written by train_n_runs.py
(result/runs_{model_type}_{dataset}_{att}.json) and runs a paired t-test
(scipy.stats.ttest_rel) per metric: AUC, F1, Recall@K, NDCG@K.

The pairing is valid because both models are trained with the same seed (555) and
the same number of runs, so run i of the proposed model and run i of the base model
see the same train/eval/test split.

Generate the inputs first, e.g.:
  python main.py --dataset music ... --model_type KGCN_LightGCN --att product            # proposed
  python main.py --dataset music ... --model_type LightGCN      --att product            # base
  # (to compare same-model_type variants, separate them with --exp_tag)

Usage (from repo root):
  python scripts/significance_test.py --dataset music \
      --proposed_model KGCN_LightGCN --base_model LightGCN --att product
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULT = os.path.join(_ROOT, "result")


def _att_tag(att, tag):
    return att + ("_" + tag if tag else "")


def load_runs(model, dataset, att, tag):
    path = os.path.join(_RESULT, "runs_{}_{}_{}.json".format(model, dataset, _att_tag(att, tag)))
    if not os.path.exists(path):
        return None, path
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def get_metric(data, name, k):
    if name in ("auc", "f1"):
        return np.array(data[name], dtype=float)
    arr = np.array(data[name], dtype=float)  # [runs, len(k_list)]
    if k not in data["k_list"]:
        return None
    return arr[:, data["k_list"].index(k)]


def stars(p):
    if p != p:  # nan
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["movie", "book", "music", "restaurant"])
    ap.add_argument("--proposed_model", default="KGCN_LightGCN")
    ap.add_argument("--base_model", default="LightGCN")
    ap.add_argument("--att", default="product", help="attention tag for both (shortcut)")
    ap.add_argument("--proposed_att", default=None)
    ap.add_argument("--base_att", default=None)
    ap.add_argument("--proposed_tag", default="")
    ap.add_argument("--base_tag", default="")
    ap.add_argument("--k", type=int, default=10, help="K for Recall@K / NDCG@K")
    ap.add_argument("--alternative", default="two-sided",
                    choices=["two-sided", "greater", "less"],
                    help="'greater' = one-sided test that proposed > base")
    args = ap.parse_args()

    p_att = args.proposed_att or args.att
    b_att = args.base_att or args.att

    prop, p_path = load_runs(args.proposed_model, args.dataset, p_att, args.proposed_tag)
    base, b_path = load_runs(args.base_model, args.dataset, b_att, args.base_tag)
    if prop is None:
        print("missing proposed runs file:", os.path.relpath(p_path, _ROOT))
        return
    if base is None:
        print("missing base runs file:", os.path.relpath(b_path, _ROOT))
        return

    metrics = [("AUC", "auc", None), ("F1", "f1", None),
               ("Recall@%d" % args.k, "recall", args.k), ("NDCG@%d" % args.k, "ndcg", args.k)]

    lines = []
    lines.append("Paired significance test -- dataset: {}".format(args.dataset))
    lines.append("proposed = {} ({})   base = {} ({})".format(
        args.proposed_model, _att_tag(p_att, args.proposed_tag),
        args.base_model, _att_tag(b_att, args.base_tag)))
    lines.append("paired t-test (scipy.ttest_rel), alternative='{}'   * p<.05  ** p<.01  *** p<.001".format(
        args.alternative))
    n_pair = min(prop["runs"], base["runs"])
    if prop["runs"] != base["runs"]:
        lines.append("WARNING: run counts differ ({} vs {}); pairing first {} runs".format(
            prop["runs"], base["runs"], n_pair))
    lines.append("runs paired = {}".format(n_pair))
    lines.append("")
    lines.append("{:<12}{:<20}{:<20}{:<12}{:<10}{:<10}{:<6}".format(
        "metric", "proposed", "base", "delta", "t", "p", "sig"))

    for label, key, k in metrics:
        if key not in prop or key not in base:
            continue
        pv = get_metric(prop, key, k)
        bv = get_metric(base, key, k)
        if pv is None or bv is None:
            continue
        pv, bv = pv[:n_pair], bv[:n_pair]
        delta = float(np.mean(pv) - np.mean(bv))
        if n_pair >= 2 and np.std(pv - bv) > 0:
            t, p = stats.ttest_rel(pv, bv, alternative=args.alternative)
        else:
            t, p = float("nan"), float("nan")
        lines.append("{:<12}{:<20}{:<20}{:<+12.4f}{:<10}{:<10}{:<6}".format(
            label,
            "{:.4f}+-{:.4f}".format(np.mean(pv), np.std(pv)),
            "{:.4f}+-{:.4f}".format(np.mean(bv), np.std(bv)),
            delta,
            "{:.3f}".format(t) if t == t else "nan",
            "{:.4f}".format(p) if p == p else "nan",
            stars(p)))

    table = "\n".join(lines)
    print(table)
    out = os.path.join(_RESULT, "significance_{}.txt".format(args.dataset))
    with open(out, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\nwritten: {}".format(os.path.relpath(out, _ROOT)))


if __name__ == "__main__":
    main()
