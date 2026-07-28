# time one music epoch: legacy vs --fast_eval (same hyperparams).
# run: python scripts/bench_fast_eval_epoch.py

import os
import sys
import time
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
PY = sys.executable

base = [
    PY, 'main.py',
    '--dataset', 'music',
    '--aggregator', 'concat',
    '--n_epochs', '1',
    '--neighbor_sample_size', '8',
    '--dim', '32',
    '--n_iter', '1',
    '--batch_size', '128',
    '--l2_weight', '1e-4',
    '--lr', '0.005',
    '--layer_size', '[32]',
    '--adj_type', 'norm',
    '--alg_type', 'lightgcn',
    '--model_type', 'KGCN_LightGCN',
    '--node_dropout', '[0.1]',
    '--mess_dropout', '[0.1]',
    '--node_dropout_flag', '1',
    '--alpha', '0.5',
    '--smoothing_steps', '8',
    '--pretrain', '0',
    '--att', 'product',
    '--runs', '1',
    '--gpu_id', '0',
    '--topk_k_list', '[2,10,50,100]',
    '--topk_user_num', '50',
]


def run(tag, extra):
    cmd = base + extra + ['--exp_tag', tag]
    print('===', tag, '===')
    t0 = time.time()
    p = subprocess.run(cmd, cwd=SRC)
    dt = time.time() - t0
    print('wall_sec=%.1f exit=%d' % (dt, p.returncode))
    return dt, p.returncode


def main():
    t_legacy, rc1 = run('bench_legacy', ['--fast_eval', '0'])
    t_fast, rc2 = run('bench_fast', ['--fast_eval', '1'])
    if rc1 == 0 and rc2 == 0 and t_fast > 0:
        print('SUMMARY legacy=%.1fs fast=%.1fs speedup=%.2fx' % (
            t_legacy, t_fast, t_legacy / t_fast))
    return 0 if rc1 == 0 and rc2 == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
