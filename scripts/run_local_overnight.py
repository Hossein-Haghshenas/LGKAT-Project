"""
Overnight sequential runner for local GPU:
1. Book Interpretation (interp_v3, ~15 min)
2. Last.fm No-KG Baseline Sparsity (base_sparsity, 3 runs, ~3 hours)
3. Book-Crossing Sparsity (proposed_sparsity, 3 runs, ~1.5 hours)
Total duration: ~4.5 - 5 hours.
"""
import os
import sys
import time
import subprocess
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
RESULT = os.path.join(ROOT, "result")
LOG_FILE = os.path.join(RESULT, "overnight_local_runs.log")

PY = sys.executable

TASKS = [
    (
        "1/3: Book Interpretation (interp_v3)",
        [
            PY, "main.py",
            "--dataset", "book",
            "--aggregator", "concat",
            "--n_epochs", "39",
            "--neighbor_sample_size", "8",
            "--dim", "64",
            "--n_iter", "1",
            "--batch_size", "256",
            "--l2_weight", "2e-5",
            "--lr", "0.0002",
            "--layer_size", "[64]",
            "--adj_type", "norm",
            "--alg_type", "lightgcn",
            "--model_type", "KGCN_LightGCN",
            "--node_dropout", "[0.1]",
            "--mess_dropout", "[0.1]",
            "--node_dropout_flag", "1",
            "--alpha", "0",
            "--smoothing_steps", "3",
            "--pretrain", "0",
            "--att", "product",
            "--runs", "1",
            "--gpu_id", "0",
            "--topk_k_list", "[2,10,50,100]",
            "--topk_user_num", "100",
            "--fusion", "gate",
            "--use_cl", "0",
            "--use_simgcl", "0",
            "--lr_decay", "0.98",
            "--early_stop", "5",
            "--early_stop_min_epoch", "0",
            "--fast_eval", "1",
            "--interpret", "1",
            "--interpret_topk", "5",
            "--log_resources", "1",
            "--exp_tag", "interp_v3",
        ]
    ),
    (
        "2/3: Last.fm No-KG Baseline Sparsity (base_sparsity, 3 runs)",
        [
            PY, "main.py",
            "--dataset", "music",
            "--aggregator", "concat",
            "--n_epochs", "39",
            "--neighbor_sample_size", "8",
            "--dim", "32",
            "--n_iter", "1",
            "--batch_size", "128",
            "--l2_weight", "1e-4",
            "--lr", "0.005",
            "--layer_size", "[32]",
            "--adj_type", "norm",
            "--alg_type", "lightgcn",
            "--model_type", "LightGCN",
            "--node_dropout", "[0.1]",
            "--mess_dropout", "[0.1]",
            "--node_dropout_flag", "1",
            "--alpha", "0.5",
            "--smoothing_steps", "8",
            "--pretrain", "0",
            "--att", "product",
            "--runs", "3",
            "--gpu_id", "0",
            "--topk_k_list", "[2,10,50,100]",
            "--topk_user_num", "0",
            "--fusion", "fixed",
            "--use_cl", "0",
            "--use_simgcl", "0",
            "--lr_decay", "0.98",
            "--early_stop", "12",
            "--early_stop_min_epoch", "25",
            "--fast_eval", "1",
            "--sparsity_eval", "1",
            "--sparsity_buckets", "[5,10,20]",
            "--log_resources", "1",
            "--exp_tag", "base_sparsity",
        ]
    ),
    (
        "3/3: Book-Crossing Sparsity (proposed_sparsity, 3 runs)",
        [
            PY, "main.py",
            "--dataset", "book",
            "--aggregator", "concat",
            "--n_epochs", "39",
            "--neighbor_sample_size", "8",
            "--dim", "64",
            "--n_iter", "1",
            "--batch_size", "256",
            "--l2_weight", "2e-5",
            "--lr", "0.0002",
            "--layer_size", "[64]",
            "--adj_type", "norm",
            "--alg_type", "lightgcn",
            "--model_type", "KGCN_LightGCN",
            "--node_dropout", "[0.1]",
            "--mess_dropout", "[0.1]",
            "--node_dropout_flag", "1",
            "--alpha", "0",
            "--smoothing_steps", "3",
            "--pretrain", "0",
            "--att", "product",
            "--runs", "3",
            "--gpu_id", "0",
            "--topk_k_list", "[2,10,50,100]",
            "--topk_user_num", "100",
            "--fusion", "gate",
            "--use_cl", "0",
            "--use_simgcl", "0",
            "--lr_decay", "0.98",
            "--early_stop", "5",
            "--early_stop_min_epoch", "0",
            "--fast_eval", "1",
            "--sparsity_eval", "1",
            "--sparsity_buckets", "[5,10,20]",
            "--log_resources", "1",
            "--exp_tag", "proposed_sparsity",
        ]
    ),
]

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def main():
    os.makedirs(RESULT, exist_ok=True)
    log(f"Starting Overnight Local Runner with {len(TASKS)} tasks...")
    overall_start = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    for i, (name, cmd) in enumerate(TASKS, 1):
        log(f"=== Starting Task {i}/{len(TASKS)}: {name} ===")
        task_start = time.time()
        try:
            # Run from src/ directory
            proc = subprocess.Popen(
                cmd,
                cwd=SRC,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )
            # Stream output to log file and stdout
            with open(LOG_FILE, "a", encoding="utf-8") as log_f:
                for line in proc.stdout:
                    log_f.write(line)
                    log_f.flush()
                    # print periodically or epoch summaries
                    if "epoch" in line.lower() or "average" in line.lower() or "best" in line.lower():
                        print(f"  {line.strip()}", flush=True)
            
            proc.wait()
            rc = proc.returncode
            elapsed = time.time() - task_start
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            time_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
            
            if rc == 0:
                log(f"=== Task {i} SUCCESS in {time_str} ===")
            else:
                log(f"=== Task {i} FAILED with return code {rc} in {time_str} ===")
        except Exception as e:
            elapsed = time.time() - task_start
            log(f"=== Task {i} EXCEPTION: {e} after {elapsed:.1f}s ===")

    total_elapsed = time.time() - overall_start
    m, s = divmod(int(total_elapsed), 60)
    h, m = divmod(m, 60)
    log(f"All overnight tasks completed in {h}h {m}m {s}s. Results ready in {RESULT}.")

if __name__ == "__main__":
    main()
