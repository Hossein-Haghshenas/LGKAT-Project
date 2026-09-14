# sequential 3-arm Book-Crossing residual-fusion SCREEN.
# frozen kg_transformer / pool=raw; --runs 5; --topk_user_num 100.
# run from repo root: conda run -n lgkat --no-capture-output python scripts/run_v2_book_residual_screen.py
from __future__ import print_function
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')

ARMS = (
    ('fixed', 'v2_book_res_fixed'),
    ('gate', 'v2_book_res_gate'),
    ('residual', 'v2_book_res_residual'),
)

COMMON = [
    sys.executable, 'main.py',
    '--dataset', 'book',
    '--aggregator', 'concat',
    '--n_epochs', '39',
    '--neighbor_sample_size', '8',
    '--dim', '64',
    '--n_iter', '1',
    '--batch_size', '256',
    '--l2_weight', '2e-5',
    '--lr', '0.0002',
    '--layer_size', '[64]',
    '--adj_type', 'norm',
    '--alg_type', 'lightgcn',
    '--model_type', 'KGCN_LightGCN',
    '--node_dropout', '[0.1]',
    '--mess_dropout', '[0.1]',
    '--node_dropout_flag', '1',
    '--alpha', '0',
    '--smoothing_steps', '3',
    '--pretrain', '0',
    '--att', 'kg_transformer',
    '--n_heads', '4',
    '--kgtr_token', 'rel_ej',
    '--kgtr_pool', 'raw',
    '--kgtr_center', '0',
    '--kgtr_ffn', '0',
    '--kgtr_rel_bias', '0',
    '--kgtr_dot', '0',
    '--runs', '5',
    '--gpu_id', '0',
    '--topk_k_list', '[2,10,50,100]',
    '--topk_user_num', '100',
    '--use_cl', '0',
    '--use_simgcl', '0',
    '--lr_decay', '0.98',
    '--early_stop', '5',
    '--early_stop_min_epoch', '0',
    '--rank_ckpt', 'r10',
    '--fast_eval', '1',
    '--log_resources', '1',
]


def main():
    os.chdir(SRC)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    for fusion, tag in ARMS:
        cmd = COMMON + ['--fusion', fusion, '--exp_tag', tag]
        print('=' * 80, flush=True)
        print('ARM fusion=%s exp_tag=%s' % (fusion, tag), flush=True)
        print(' '.join(cmd), flush=True)
        print('=' * 80, flush=True)
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            print('ARM_FAILED fusion=%s rc=%s' % (fusion, rc), flush=True)
            sys.exit(rc)
        print('ARM_DONE fusion=%s' % fusion, flush=True)
    print('ALL_ARMS_DONE', flush=True)


if __name__ == '__main__':
    main()
