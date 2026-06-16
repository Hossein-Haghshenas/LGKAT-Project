"""
Task 5: hyperparameter sensitivity sweep harness.

For a chosen dataset it varies ONE hyperparameter at a time (others fixed at the
paper/README config for that dataset) over a grid, runs the finalized multi-run
training for each value, and writes a machine-readable CSV with mean+-std of
AUC / F1 / Recall@10 / NDCG@10 -- ready to plot "metric vs hyperparameter".

Swept parameters (paper Section 4-4-2):
  smoothing_steps (S), dim, neighbor_sample_size, alpha (alpha-init), cl_lambda.

Each sweep point is a full training run, so this is compute-heavy. Two modes:
  --dry_run : print the exact `python main.py ...` commands and exit (run them yourself)
  (default) : execute each run, parse its stdout, and append a row to the CSV

Examples (from repo root):
  python scripts/sensitivity_sweep.py --dataset music --param smoothing_steps --dry_run
  python scripts/sensitivity_sweep.py --dataset music --param all --dry_run
  python scripts/sensitivity_sweep.py --dataset music --param dim --values 8,16,32,64 --runs 3

Output: result/sensitivity_<dataset>_<param>.csv
"""
import argparse
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
_RESULT = os.path.join(_ROOT, "result")

# Per-dataset base configuration (mirrors the README reproduction commands). The
# swept parameter overrides its key for each grid point; everything else is fixed.
BASE = {
    "music":      dict(neighbor_sample_size=8, dim=32, n_iter=1, batch_size=128,
                       l2_weight="1e-4", lr=0.005, layer_size="[32]", adj_type="norm",
                       alpha=0.5, smoothing_steps=8),
    "book":       dict(neighbor_sample_size=8, dim=64, n_iter=1, batch_size=256,
                       l2_weight="2e-5", lr="5e-5", layer_size="[64]", adj_type="norm",
                       alpha=0.0, smoothing_steps=3),
    "restaurant": dict(neighbor_sample_size=4, dim=8, n_iter=2, batch_size=65536,
                       l2_weight="1e-7", lr="2e-2", layer_size="[8]", adj_type="norm",
                       alpha=0.5, smoothing_steps=1),
    "movie":      dict(neighbor_sample_size=4, dim=32, n_iter=2, batch_size=65536,
                       l2_weight="5e-6", lr="2e-2", layer_size="[32]", adj_type="plain",
                       alpha=0.0, smoothing_steps=1),
}

# Shared (dataset-independent) arguments.
COMMON = dict(aggregator="concat", alg_type="lightgcn", model_type="KGCN_LightGCN",
              node_dropout="[0.1]", mess_dropout="[0.1]", node_dropout_flag=1,
              agg_type="weighted_avg", pretrain=0, att="product",
              topk_k_list="[2,10,50,100]", topk_user_num=100, gpu_id=0, report_ndcg=1)

DEFAULT_GRID = {
    "smoothing_steps":      [1, 2, 3, 4, 6, 8],
    "dim":                  [8, 16, 32, 64],
    "neighbor_sample_size": [2, 4, 8, 16],
    "alpha":                [0.0, 0.25, 0.5, 0.75, 1.0],
    "cl_lambda":            [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
}
SWEEP_PARAMS = list(DEFAULT_GRID.keys())


def build_args(dataset, param, value, runs, n_epochs, att, topk_user_num):
    cfg = dict(COMMON)
    cfg.update(BASE[dataset])
    cfg["dataset"] = dataset
    cfg["runs"] = runs
    cfg["n_epochs"] = n_epochs
    cfg["att"] = att
    cfg["topk_user_num"] = topk_user_num

    # cl_lambda only has an effect with the contrastive module switched on
    if param == "cl_lambda":
        cfg["use_cl"] = 1

    cfg[param] = value
    # dim is the embedding width; keep the LightGCN layer width consistent with it
    if param == "dim":
        cfg["layer_size"] = "[{}]".format(value)

    # tag result filenames so per-point logs don't overwrite each other
    cfg["exp_tag"] = "sweep_{}_{}".format(param, value)

    argv = []
    for k, v in cfg.items():
        argv += ["--{}".format(k), str(v)]
    return argv


_FLOAT = r"[0-9]*\.?[0-9]+"


def parse_metrics(stdout):
    out = {"auc_mean": float("nan"), "auc_std": float("nan"),
           "f1_mean": float("nan"), "f1_std": float("nan"),
           "recall@10_mean": float("nan"), "recall@10_std": float("nan"),
           "ndcg@10_mean": float("nan"), "ndcg@10_std": float("nan")}
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("average auc:"):
            nums = re.findall(_FLOAT, s.split(":", 1)[-1])  # numbers after the label only
            if len(nums) >= 2:
                out["auc_mean"], out["auc_std"] = float(nums[0]), float(nums[1])
        elif s.startswith("average f1"):
            nums = re.findall(_FLOAT, s.split(":", 1)[-1])  # avoid capturing the '1' in 'f1'
            if len(nums) >= 2:
                out["f1_mean"], out["f1_std"] = float(nums[0]), float(nums[1])
        elif s.startswith("R@10 "):  # paper_style block line
            rec = re.search(r"recall:\s*({f})\+-({f})".format(f=_FLOAT), s)
            ndc = re.search(r"ndcg:\s*({f})\+-({f})".format(f=_FLOAT), s)
            if rec:
                out["recall@10_mean"], out["recall@10_std"] = float(rec.group(1)), float(rec.group(2))
            if ndc:
                out["ndcg@10_mean"], out["ndcg@10_std"] = float(ndc.group(1)), float(ndc.group(2))
    return out


def run_point(python_exe, argv):
    proc = subprocess.run([python_exe, "main.py"] + argv, cwd=_SRC,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)
    return proc.returncode, proc.stdout


COLUMNS = ["dataset", "param", "value", "runs", "n_epochs",
           "auc_mean", "auc_std", "f1_mean", "f1_std",
           "recall@10_mean", "recall@10_std", "ndcg@10_mean", "ndcg@10_std"]


def sweep_one(args, param, python_exe):
    values = [float(v) if ("." in v or param in ("alpha", "cl_lambda")) else int(v)
              for v in args.values.split(",")] if args.values else DEFAULT_GRID[param]

    out_csv = os.path.join(_RESULT, "sensitivity_{}_{}.csv".format(args.dataset, param))

    if args.dry_run:
        print("# {} sweep -- {} = {}".format(args.dataset, param, values))
        for v in values:
            argv = build_args(args.dataset, param, v, args.runs, args.n_epochs,
                              args.att, args.topk_user_num)
            print("python main.py " + " ".join(argv))
        print("# -> would write {}\n".format(os.path.relpath(out_csv, _ROOT)))
        return

    if not os.path.exists(_RESULT):
        os.makedirs(_RESULT)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(COLUMNS) + "\n")
    print("sweeping {} over {} -> {}".format(param, values, os.path.relpath(out_csv, _ROOT)))

    for v in values:
        argv = build_args(args.dataset, param, v, args.runs, args.n_epochs,
                          args.att, args.topk_user_num)
        print("  [{} = {}] running ...".format(param, v), flush=True)
        code, stdout = run_point(python_exe, argv)
        if code != 0:
            print("    FAILED (exit {}); last output:\n{}".format(code, stdout[-500:]))
            m = parse_metrics("")
        else:
            m = parse_metrics(stdout)
            print("    auc={:.4f} f1={:.4f} R@10={:.4f} N@10={:.4f}".format(
                m["auc_mean"], m["f1_mean"], m["recall@10_mean"], m["ndcg@10_mean"]))
        row = [args.dataset, param, v, args.runs, args.n_epochs,
               m["auc_mean"], m["auc_std"], m["f1_mean"], m["f1_std"],
               m["recall@10_mean"], m["recall@10_std"], m["ndcg@10_mean"], m["ndcg@10_std"]]
        with open(out_csv, "a", encoding="utf-8") as f:  # append per point: partial progress survives
            f.write(",".join(str(x) for x in row) + "\n")
    print("done -> {}".format(os.path.relpath(out_csv, _ROOT)))


def main():
    ap = argparse.ArgumentParser(description="LGKAT hyperparameter sensitivity sweep")
    ap.add_argument("--dataset", required=True, choices=list(BASE.keys()))
    ap.add_argument("--param", required=True, choices=SWEEP_PARAMS + ["all"])
    ap.add_argument("--values", default="", help="comma list overriding the default grid (single param only)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n_epochs", type=int, default=39)
    ap.add_argument("--att", default="product")
    ap.add_argument("--topk_user_num", type=int, default=100)
    ap.add_argument("--dry_run", action="store_true", help="print commands instead of running them")
    ap.add_argument("--python", default=sys.executable, help="python executable for the training subprocess")
    args = ap.parse_args()

    params = SWEEP_PARAMS if args.param == "all" else [args.param]
    if args.param == "all" and args.values:
        ap.error("--values cannot be combined with --param all")

    for p in params:
        sweep_one(args, p, args.python)


if __name__ == "__main__":
    main()
