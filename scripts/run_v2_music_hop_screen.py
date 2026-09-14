# sequential 2-arm Last.fm n_iter screen (locked V2 topology hypothesis).
# frozen kg_transformer / pool=raw / gate + CL + SimGCL.
# only --n_iter changes: 1 vs 2. --runs 5 --topk_user_num 100.
# run from repo root: conda run -n lgkat --no-capture-output python scripts/run_v2_music_hop_screen.py
from __future__ import print_function
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
DATA = os.path.join(ROOT, 'data', 'music')

ARMS = (
    (1, 'v2_music_hop1'),
    (2, 'v2_music_hop2'),
)

COMMON = [
    sys.executable, 'main.py',
    '--dataset', 'music',
    '--aggregator', 'concat',
    '--n_epochs', '39',
    '--neighbor_sample_size', '8',
    '--dim', '32',
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
    '--topk_k_list', '[2,10,50,100]',
    '--topk_user_num', '100',
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
]


def _load_npy_or_txt(path_no_ext):
    npy = path_no_ext + '.npy'
    txt = path_no_ext + '.txt'
    if os.path.exists(npy):
        return np.load(npy)
    return np.loadtxt(txt, dtype=np.int64)


def _construct_adj(kg, n_entity, n_neighbor, seed_after_split):
    # replicate construct_adj RNG after dataset_split reseeded to 555 and drew eval/test.
    np.random.seed(555)
    n_ratings = seed_after_split
    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * 0.2), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    np.random.choice(list(left), size=int(n_ratings * 0.2), replace=False)
    adj_entity = np.zeros([n_entity, n_neighbor], dtype=np.int64)
    for entity in range(n_entity):
        neighbors = kg[entity]
        n_nb = len(neighbors)
        if n_nb >= n_neighbor:
            sampled = np.random.choice(list(range(n_nb)), size=n_neighbor, replace=False)
        else:
            sampled = np.random.choice(list(range(n_nb)), size=n_neighbor, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled])
    return adj_entity


def preflight():
    ratings = _load_npy_or_txt(os.path.join(DATA, 'ratings_final'))
    kg_np = _load_npy_or_txt(os.path.join(DATA, 'kg_final'))
    n_item = len(set(ratings[:, 1].tolist()))
    n_entity = len(set(kg_np[:, 0].tolist()) | set(kg_np[:, 2].tolist()))
    kg = defaultdict(list)
    for h, r, t in kg_np:
        kg[int(h)].append((int(t), int(r)))
        kg[int(t)].append((int(h), int(r)))
    for e in range(n_entity):
        kg.setdefault(e, [])
    adj_a = _construct_adj(kg, n_entity, 8, ratings.shape[0])
    adj_b = _construct_adj(kg, n_entity, 8, ratings.shape[0])
    same = bool(np.array_equal(adj_a, adj_b))
    print('PREFLIGHT construct_adj identical across two draws: %s' % same, flush=True)
    print('PREFLIGHT construct_adj ignores n_iter (not an argument)', flush=True)

    n1_items = []
    n2_items = []
    gain_only_h2 = 0
    for i in range(n_item):
        hop1 = set(int(x) for x in adj_a[i].tolist() if x != i)
        hop1_items = set(x for x in hop1 if 0 <= x < n_item)
        hop2 = set()
        for v in adj_a[i].tolist():
            for w in adj_a[int(v)].tolist():
                if w != i:
                    hop2.add(int(w))
        hop2_items = set(x for x in hop2 if 0 <= x < n_item)
        n1_items.append(len(hop1_items))
        n2_items.append(len(hop2_items))
        if len(hop1_items) == 0 and len(hop2_items) >= 1:
            gain_only_h2 += 1
    n1_items = np.array(n1_items, dtype=float)
    n2_items = np.array(n2_items, dtype=float)
    print('PREFLIGHT sampled unique other catalog items hop1: mean=%.4f median=%.1f pct_ge1=%.4f' % (
        n1_items.mean(), np.median(n1_items), (n1_items >= 1).mean()), flush=True)
    print('PREFLIGHT sampled unique other catalog items hop2: mean=%.4f median=%.1f pct_ge1=%.4f' % (
        n2_items.mean(), np.median(n2_items), (n2_items >= 1).mean()), flush=True)
    print('PREFLIGHT fraction of items with first item-item neighbor only at hop2: %.4f' % (
        gain_only_h2 / float(n_item)), flush=True)
    if not same:
        sys.exit(2)


def main():
    preflight()
    os.chdir(SRC)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    # identical list(set) order in dataset_split so both arms share the same split/adj RNG
    env['PYTHONHASHSEED'] = '0'
    for n_iter, tag in ARMS:
        cmd = COMMON + ['--n_iter', str(n_iter), '--exp_tag', tag]
        print('=' * 80, flush=True)
        print('ARM n_iter=%s exp_tag=%s' % (n_iter, tag), flush=True)
        print(' '.join(cmd), flush=True)
        print('=' * 80, flush=True)
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            print('ARM_FAILED n_iter=%s rc=%s' % (n_iter, rc), flush=True)
            sys.exit(rc)
        print('ARM_DONE n_iter=%s' % n_iter, flush=True)
    print('ALL_ARMS_DONE', flush=True)


if __name__ == '__main__':
    main()
