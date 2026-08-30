"""Aggregate valid per-run JSON from --log_resources 5-run SSL experiments.

Does not invent metrics, time, or VRAM. Refuses a 5-run mean unless enough
status==ok runs exist. Legacy JSONs without run_records are treated as all-valid
for AUC/F1/Recall only (time/VRAM reported as missing).

Usage (from repo root):
  python scripts/aggregate_final_runs.py --exp_tag paper_ssl_final_5run
  python scripts/aggregate_final_runs.py --exp_tag paper_ssl_final_5run --datasets music,book --min_runs 5
"""
from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULT = os.path.join(_ROOT, "result")
_K_PAPER = (2, 10, 50, 100)


def _att_tag(att, tag):
    return att + ("_" + tag if tag else "")


def _json_path(dataset, att, tag):
    return os.path.join(
        _RESULT, "runs_KGCN_LightGCN_{}_{}.json".format(dataset, _att_tag(att, tag)))


def _mean_std(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, 0
    return float(np.mean(arr)), float(np.std(arr)), int(arr.size)


def _fmt_ms(mean, std, n):
    if mean is None:
        return "unavailable (n=0)"
    return "{:.4f} ± {:.4f}  (n={})".format(mean, std, n)


def _fmt_res(mean, std, n, unit):
    if mean is None:
        return "unavailable"
    return "{:.1f} ± {:.1f} {}  (n={})".format(mean, std, unit, n)


def _recall_at(rec, k_list, k):
    recall = rec.get("recall")
    if recall is None or k not in k_list:
        return None
    return recall[k_list.index(k)]


def _valid_records(data):
    records = data.get("run_records")
    k_list = list(data.get("k_list") or [])
    if records:
        ok = [r for r in records if r.get("status") == "ok"]
        crashed = [r for r in records if r.get("status") != "ok"]
        return ok, crashed, k_list, True
    # legacy json: arrays only, all rows treated as valid, no time/vram
    n = int(data.get("runs") or 0)
    ok = []
    for i in range(n):
        rec = {
            "run_index": i,
            "status": "ok",
            "auc": data["auc"][i] if i < len(data.get("auc") or []) else None,
            "f1": data["f1"][i] if i < len(data.get("f1") or []) else None,
            "recall": data["recall"][i] if i < len(data.get("recall") or []) else None,
            "train_seconds": None,
            "total_seconds": None,
            "peak_vram_mb": None,
        }
        ok.append(rec)
    return ok, [], k_list, False


def _summarize_dataset(dataset, att, tag, min_runs):
    path = _json_path(dataset, att, tag)
    rel = os.path.relpath(path, _ROOT)
    out = {
        "dataset": dataset,
        "path": rel.replace("\\", "/"),
        "ok": False,
        "errors": [],
    }
    if not os.path.exists(path):
        out["errors"].append("missing json")
        return out
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    ok, crashed, k_list, has_records = _valid_records(data)
    out["n_attempted"] = int(data.get("n_attempted") or data.get("runs") or (len(ok) + len(crashed)))
    out["n_valid"] = len(ok)
    out["n_excluded"] = len(crashed)
    out["has_run_records"] = has_records
    out["excluded"] = [
        {"run_index": r.get("run_index"), "status": r.get("status"), "error": r.get("error")}
        for r in crashed
    ]

    if out["n_valid"] < min_runs:
        out["errors"].append(
            "need {} valid runs, have {} (excluded {})".format(
                min_runs, out["n_valid"], out["n_excluded"]))
        return out

    metrics = {}
    auc_m, auc_s, auc_n = _mean_std([r["auc"] for r in ok])
    f1_m, f1_s, f1_n = _mean_std([r["f1"] for r in ok])
    metrics["AUC"] = {"mean": auc_m, "std": auc_s, "n": auc_n, "text": _fmt_ms(auc_m, auc_s, auc_n)}
    metrics["F1"] = {"mean": f1_m, "std": f1_s, "n": f1_n, "text": _fmt_ms(f1_m, f1_s, f1_n)}
    for k in _K_PAPER:
        vals = [_recall_at(r, k_list, k) for r in ok]
        m, s, n = _mean_std(vals)
        metrics["Recall@{}".format(k)] = {
            "mean": m, "std": s, "n": n, "text": _fmt_ms(m, s, n),
        }

    tr_m, tr_s, tr_n = _mean_std([r.get("train_seconds") for r in ok])
    tot_m, tot_s, tot_n = _mean_std([r.get("total_seconds") for r in ok])
    vr_m, vr_s, vr_n = _mean_std([r.get("peak_vram_mb") for r in ok])
    resources = {
        "train_seconds": {"mean": tr_m, "std": tr_s, "n": tr_n, "text": _fmt_res(tr_m, tr_s, tr_n, "s")},
        "total_seconds": {"mean": tot_m, "std": tot_s, "n": tot_n, "text": _fmt_res(tot_m, tot_s, tot_n, "s")},
        "peak_vram_mb": {"mean": vr_m, "std": vr_s, "n": vr_n, "text": _fmt_res(vr_m, vr_s, vr_n, "MiB")},
    }

    out["ok"] = True
    out["metrics"] = metrics
    out["resources"] = resources
    out["per_run"] = [
        {
            "run_index": r.get("run_index"),
            "seed_eval": r.get("seed_eval"),
            "seed_test": r.get("seed_test"),
            "auc": r.get("auc"),
            "f1": r.get("f1"),
            "recall": r.get("recall"),
            "train_seconds": r.get("train_seconds"),
            "total_seconds": r.get("total_seconds"),
            "peak_vram_mb": r.get("peak_vram_mb"),
            "vram_source": r.get("vram_source"),
            "status": r.get("status"),
        }
        for r in ok
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--att", default="product")
    ap.add_argument("--datasets", default="music,book,movie,restaurant")
    ap.add_argument("--min_runs", type=int, default=5)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    summary = {
        "exp_tag": args.exp_tag,
        "att": args.att,
        "min_runs": args.min_runs,
        "datasets": {},
        "all_ok": True,
    }

    lines = []
    lines.append("LGKAT-SSL final aggregation  exp_tag={}  att={}  min_runs={}".format(
        args.exp_tag, args.att, args.min_runs))
    lines.append("mean ± std uses np.std (population, same as train_n_runs.py)")
    lines.append("")

    for ds in datasets:
        block = _summarize_dataset(ds, args.att, args.exp_tag, args.min_runs)
        summary["datasets"][ds] = block
        lines.append("=== {} ===".format(ds))
        lines.append("file: {}".format(block["path"]))
        if not block["ok"]:
            summary["all_ok"] = False
            for err in block["errors"]:
                lines.append("FAIL: {}".format(err))
            if block.get("excluded"):
                lines.append("excluded: {}".format(block["excluded"]))
            lines.append("")
            continue
        lines.append("valid runs: {}/{}  excluded: {}".format(
            block["n_valid"], block["n_attempted"], block["n_excluded"]))
        if not block["has_run_records"]:
            lines.append("note: legacy json (no run_records); time/VRAM not recorded")
        for name in ["AUC", "F1", "Recall@2", "Recall@10", "Recall@50", "Recall@100"]:
            lines.append("  {:<12} {}".format(name, block["metrics"][name]["text"]))
        lines.append("  {:<12} {}".format("train_s", block["resources"]["train_seconds"]["text"]))
        lines.append("  {:<12} {}".format("total_s", block["resources"]["total_seconds"]["text"]))
        lines.append("  {:<12} {}".format("peak_vram", block["resources"]["peak_vram_mb"]["text"]))
        lines.append("")

    text = "\n".join(lines)
    print(text)
    out_name = "summary_{}.json".format(args.exp_tag)
    out_path = os.path.join(_RESULT, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    txt_path = os.path.join(_RESULT, "summary_{}.txt".format(args.exp_tag))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("written: {}".format(os.path.relpath(out_path, _ROOT)))
    print("written: {}".format(os.path.relpath(txt_path, _ROOT)))
    if not summary["all_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
