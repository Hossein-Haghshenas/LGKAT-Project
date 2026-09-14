# sequential paper-protocol ranking confirmation (no new architecture).
# frozen kg_transformer / pool=raw; Product same-stack control.
# only change vs 100-user champions: --topk_user_num 0 --runs 5.
# exp_tag protocol_allusers does not overwrite 100-user files.
# default arms: music kgtr, music Product, book kgtr.
# movie/restaurant are opt-in (--only movie_kgtr / restaurant_kgtr).
# run from repo root:
#   conda run -n lgkat --no-capture-output python scripts/run_protocol_allusers.py
# music pair only:
#   conda run -n lgkat --no-capture-output python scripts/run_protocol_allusers.py --only music_kgtr music_product
from __future__ import print_function
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')

PY = sys.executable
TAG = 'protocol_allusers'
K_LIST = '[2,10,50,100]'

MUSIC_KGTR = [
    PY, 'main.py',
    '--dataset', 'music',
    '--aggregator', 'concat',
    '--n_epochs', '39',
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
    '--topk_k_list', K_LIST,
    '--topk_user_num', '0',
    '--fusion', 'gate',
    '--use_cl', '1',
    '--cl_lambda', '0.02',
    '--cl_tau', '0.2',
    '--use_simgcl', '1',
    '--simgcl_lambda', '0.02',
    '--simgcl_eps', '0.05',
    '--lr_decay', '0.98',
    '--early_stop', '12',
    '--early_stop_min_epoch', '25',
    '--rank_ckpt', 'r10',
    '--fast_eval', '1',
    '--log_resources', '1',
    '--exp_tag', TAG,
]

MUSIC_PRODUCT = [
    PY, 'main.py',
    '--dataset', 'music',
    '--aggregator', 'concat',
    '--n_epochs', '39',
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
    '--runs', '5',
    '--gpu_id', '0',
    '--topk_k_list', K_LIST,
    '--topk_user_num', '0',
    '--fusion', 'gate',
    '--use_cl', '1',
    '--cl_lambda', '0.02',
    '--cl_tau', '0.2',
    '--use_simgcl', '1',
    '--simgcl_lambda', '0.02',
    '--simgcl_eps', '0.05',
    '--lr_decay', '0.98',
    '--early_stop', '12',
    '--rank_ckpt', 'r10',
    '--fast_eval', '1',
    '--log_resources', '1',
    '--exp_tag', TAG,
]

BOOK_KGTR = [
    PY, 'main.py',
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
    '--topk_k_list', K_LIST,
    '--topk_user_num', '0',
    '--fusion', 'gate',
    '--use_cl', '0',
    '--use_simgcl', '0',
    '--lr_decay', '0.98',
    '--early_stop', '5',
    '--early_stop_min_epoch', '0',
    '--rank_ckpt', 'r10',
    '--fast_eval', '1',
    '--log_resources', '1',
    '--exp_tag', TAG,
]

MOVIE_KGTR = [
    PY, 'main.py',
    '--dataset', 'movie',
    '--aggregator', 'concat',
    '--n_epochs', '39',
    '--neighbor_sample_size', '4',
    '--dim', '32',
    '--n_iter', '2',
    '--batch_size', '8192',
    '--l2_weight', '5e-6',
    '--lr', '0.0025',
    '--layer_size', '[32]',
    '--adj_type', 'plain',
    '--alg_type', 'lightgcn',
    '--model_type', 'KGCN_LightGCN',
    '--node_dropout', '[0.1]',
    '--mess_dropout', '[0.1]',
    '--node_dropout_flag', '1',
    '--alpha', '0',
    '--smoothing_steps', '1',
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
    '--topk_k_list', K_LIST,
    '--topk_user_num', '0',
    '--fusion', 'fixed',
    '--use_cl', '0',
    '--use_simgcl', '0',
    '--lr_decay', '0.98',
    '--early_stop', '8',
    '--early_stop_min_epoch', '0',
    '--rank_ckpt', 'r10',
    '--fast_eval', '1',
    '--log_resources', '1',
    '--exp_tag', TAG,
]

RESTAURANT_KGTR = [
    PY, 'main.py',
    '--dataset', 'restaurant',
    '--aggregator', 'concat',
    '--n_epochs', '39',
    '--neighbor_sample_size', '4',
    '--dim', '8',
    '--n_iter', '2',
    '--batch_size', '32768',
    '--l2_weight', '1e-7',
    '--lr', '0.02',
    '--layer_size', '[8]',
    '--adj_type', 'norm',
    '--alg_type', 'lightgcn',
    '--model_type', 'KGCN_LightGCN',
    '--node_dropout', '[0.1]',
    '--mess_dropout', '[0.1]',
    '--node_dropout_flag', '1',
    '--agg_type', 'weighted_avg',
    '--smoothing_steps', '1',
    '--pretrain', '0',
    '--alpha', '0.5',
    '--att', 'kg_transformer',
    '--n_heads', '2',
    '--kgtr_token', 'rel_ej',
    '--kgtr_pool', 'raw',
    '--kgtr_center', '0',
    '--kgtr_ffn', '0',
    '--kgtr_rel_bias', '0',
    '--kgtr_dot', '0',
    '--runs', '5',
    '--gpu_id', '0',
    '--topk_k_list', K_LIST,
    '--topk_user_num', '0',
    '--fusion', 'fixed',
    '--use_cl', '0',
    '--use_simgcl', '0',
    '--lr_decay', '0.98',
    '--early_stop', '5',
    '--early_stop_min_epoch', '0',
    '--rank_ckpt', 'r10',
    '--fast_eval', '1',
    '--log_resources', '1',
    '--exp_tag', TAG,
]

ARMS = (
    ('music_kgtr', MUSIC_KGTR),
    ('music_product', MUSIC_PRODUCT),
    ('book_kgtr', BOOK_KGTR),
    ('movie_kgtr', MOVIE_KGTR),
    ('restaurant_kgtr', RESTAURANT_KGTR),
)

ARM_IDS = tuple(name for name, _ in ARMS)
DEFAULT_ARMS = ('music_kgtr', 'music_product', 'book_kgtr')
SLOW_ARMS = ('movie_kgtr', 'restaurant_kgtr')


def _selected(only):
    if not only:
        return list(DEFAULT_ARMS)
    unknown = [x for x in only if x not in ARM_IDS]
    if unknown:
        sys.exit('unknown arm(s): %s  known: %s' % (
            ' '.join(unknown), ' '.join(ARM_IDS)))
    return list(only)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--only', nargs='+', default=None, metavar='ARM',
        help='subset of: %s (default: %s)' % (
            ' '.join(ARM_IDS), ' '.join(DEFAULT_ARMS)))
    args = ap.parse_args()
    selected = _selected(args.only)
    slow = [a for a in selected if a in SLOW_ARMS]
    if slow:
        print('WARNING slow all-users arm(s): %s' % ' '.join(slow), flush=True)

    os.chdir(SRC)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONHASHSEED'] = '0'
    lookup = dict(ARMS)
    for name in selected:
        cmd = lookup[name]
        print('=' * 80, flush=True)
        print('ARM %s exp_tag=%s topk_user_num=0 runs=5' % (name, TAG), flush=True)
        print(' '.join(cmd), flush=True)
        print('=' * 80, flush=True)
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            print('ARM_FAILED %s rc=%s' % (name, rc), flush=True)
            sys.exit(rc)
        print('ARM_DONE %s' % name, flush=True)
    print('ALL_ARMS_DONE', flush=True)


if __name__ == '__main__':
    main()
