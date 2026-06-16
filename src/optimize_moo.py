"""
optimize_moo.py
===============
MOKAR driver: Multi-Objective Knowledge-Aware Recommendation via NSGA-II.

It wraps the existing LGKAT model (model_type='KGCN_LightGCN') in a multi-
objective evolutionary search over:
  * hyper-parameters  (dim, #LightGCN layers, #KG hops, neighbour sample size,
                       learning rate, L2 weight, CF/KG balance alpha,
                       neighbour aggregator, KG attention type)
  * a binary KG RELATION-SELECTION mask (which relations to keep)

Objectives (all MINIMIZED internally; accuracy & diversity are negated):
  1. accuracy   = validation AUC                      (higher is better)
  2. diversity  = catalog coverage@10                 (higher is better)
  3. efficiency = number of trainable parameters      (lower  is better)

Why this is expected to IMPROVE the model, not just reshuffle it:
  * the hyper-parameters of the base paper were hand-tuned per dataset; a guided
    search over a richer space routinely finds better operating points;
  * relation selection prunes noisy / uninformative KG relations, which directly
    addresses the base paper's stated limitation (sensitivity to KG quality) and
    can raise accuracy while shrinking the model;
  * returning a Pareto front (instead of one model) lets a practitioner pick the
    accuracy/diversity/efficiency trade-off they need.

CRITICAL correctness choice: the train/eval/test split and the user-item graph
are built ONCE and reused across every candidate; only the KG adjacency is
rebuilt per genome (it depends on the relation mask and neighbour sample size).
This keeps every candidate comparable. Contrastive learning is disabled
(cl_weight=0) so this stays pure LGKAT and distinct from the earlier LGKAT-SSL.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import tensorflow as tf

from data_loader import load_rating, construct_kg, construct_adj
from model import KGCN
from train_n_runs import get_feed_dict, ctr_eval, topk_settings
from objectives import (catalog_coverage, count_trainable_params,
                        hypervolume, pareto_front_indices, knee_point_index)
from relation_select import n_active
from moo_core import run_nsga2


# Per-dataset training batch size (the model needs full batches in its reshape
# ops, so this is fixed rather than searched).
DATASET_BATCH = {'music': 128, 'book': 256, 'restaurant': 4096, 'movie': 8192}


# --------------------------------------------------------------------------- #
# Data: load ratings + KG triples ONCE
# --------------------------------------------------------------------------- #
def _load_static_data(dataset, seed, adj_type='norm', ratio=1.0):
    """Load ratings (fixed split + user-item graph) and raw KG triples once."""
    rating_args = argparse.Namespace(dataset=dataset, seed=seed, ratio=ratio,
                                     adj_type=adj_type)
    n_user, n_item, train_data, eval_data, test_data, user_item_adj = load_rating(rating_args)

    kg_file = '../data/' + dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    return dict(n_user=n_user, n_item=n_item, n_entity=n_entity, n_relation=n_relation,
                train=train_data, eval=eval_data, test=test_data,
                ui_adj=user_item_adj, kg_np=kg_np)


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def build_space(n_relation):
    return [
        {'name': 'dim',        'type': 'cat', 'choices': [8, 16, 32, 64]},
        {'name': 'n_layers',   'type': 'cat', 'choices': [1, 2, 3, 4]},
        {'name': 'n_iter',     'type': 'cat', 'choices': [1, 2, 3]},
        {'name': 'nss',        'type': 'cat', 'choices': [4, 8, 16]},
        {'name': 'lr',         'type': 'float', 'low': 1e-4, 'high': 5e-2, 'log': True},
        {'name': 'l2',         'type': 'float', 'low': 1e-7, 'high': 1e-4, 'log': True},
        {'name': 'alpha',      'type': 'float', 'low': 0.0, 'high': 1.0},
        {'name': 'aggregator', 'type': 'cat', 'choices': ['sum', 'concat', 'neighbor', 'graphsage']},
        {'name': 'att',        'type': 'cat', 'choices': ['u_r', 'h_ur_t', 'u_r_mlp', 'u_h_r_t']},
        {'name': 'rmask',      'type': 'binary', 'length': int(n_relation), 'min_active': 1, 'p_keep': 0.7},
    ]


def _args_from_genome(genome, dataset, batch_size, eval_epochs, seed):
    """Turn a genome into a full argparse.Namespace the KGCN model understands."""
    dim = int(genome['dim'])
    n_layers = int(genome['n_layers'])
    return argparse.Namespace(
        dataset=dataset,
        # --- searched ---
        dim=dim,
        n_iter=int(genome['n_iter']),
        neighbor_sample_size=int(genome['nss']),
        lr=float(genome['lr']),
        l2_weight=float(genome['l2']),
        alpha=float(genome['alpha']),
        aggregator=str(genome['aggregator']),
        att=str(genome['att']),
        layer_size=str([dim] * n_layers),
        node_dropout=str([0.1] * n_layers),
        mess_dropout=str([0.1] * n_layers),
        # --- fixed: LGKAT (LightGCN + KG attention), no contrastive learning ---
        model_type='KGCN_LightGCN',
        alg_type='lightgcn',
        agg_type='weighted_avg',
        adj_type='norm',
        node_dropout_flag=0,
        cl_weight=0.0,
        cl_temperature=1.0,
        ls_weight=0,
        smoothing_steps=1,
        batch_size=int(batch_size),
        n_epochs=int(eval_epochs),
        runs=1,
        ratio=1,
        gpu_id=0,
        pretrain=0,
        logging='save',
        seed=int(seed),
        params='[]',
    )


# --------------------------------------------------------------------------- #
# Coverage helper (top-K recommended items per sampled user)
# --------------------------------------------------------------------------- #
def _recommended_items(args, sess, model, user_list, train_record, item_set, k, batch_size):
    n_drop = len(eval(args.layer_size))
    zeros = [0.] * n_drop
    rec = []
    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        if len(test_item_list) == 0:
            continue
        item_score = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {
                model.user_indices: [user] * batch_size,
                model.item_indices: test_item_list[start:start + batch_size],
                model.node_dropout: zeros, model.mess_dropout: zeros})
            for it, sc in zip(items, scores):
                item_score[it] = sc
            start += batch_size
        if start < len(test_item_list):
            pad = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)
            items, scores = model.get_scores(sess, {
                model.user_indices: [user] * batch_size,
                model.item_indices: pad,
                model.node_dropout: zeros, model.mess_dropout: zeros})
            for it, sc in zip(items, scores):
                item_score[it] = sc
        topk = [i for i, _ in sorted(item_score.items(), key=lambda x: x[1], reverse=True)[:k]]
        rec.append(topk)
    return rec


# --------------------------------------------------------------------------- #
# Single-candidate evaluation
# --------------------------------------------------------------------------- #
def make_evaluator(data, dataset, batch_size, eval_epochs, seed, topk_k=10, verbose=False):
    cache = {}

    def _key(genome):
        return (genome['dim'], genome['n_layers'], genome['n_iter'], genome['nss'],
                round(float(genome['lr']), 6), round(float(genome['l2']), 9),
                round(float(genome['alpha']), 4), genome['aggregator'], genome['att'],
                tuple(int(x) for x in genome['rmask']))

    def evaluate(genome):
        key = _key(genome)
        if key in cache:
            return cache[key]

        args = _args_from_genome(genome, dataset, batch_size, eval_epochs, seed)
        mask = np.asarray(genome['rmask'])
        keep = set(int(r) for r in np.where(mask > 0)[0])

        # Rebuild ONLY the KG adjacency for this relation subset (deterministic).
        kg = construct_kg(data['kg_np'], keep_relations=keep)
        adj_entity, adj_relation = construct_adj(kg, data['n_entity'],
                                                 args.neighbor_sample_size, seed=seed)

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(seed)
        model = KGCN(args, data['n_user'], data['n_item'], data['n_entity'],
                     data['n_relation'], adj_entity, adj_relation, data['ui_adj'],
                     kg=None, pretrain=None)
        n_params = count_trainable_params()

        train_data, eval_data = data['train'], data['eval']
        bs = args.batch_size
        user_list, train_record, _, item_set, _ = topk_settings(
            True, train_data, eval_data, data['n_item'], seed=seed)

        best_auc, best_cov = 0.0, 0.0
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        rng = np.random.RandomState(seed)
        try:
            with tf.compat.v1.Session(config=config) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                for step in range(args.n_epochs):
                    order = np.arange(train_data.shape[0])
                    rng.shuffle(order)
                    td = train_data[order]
                    start = 0
                    while start + bs <= td.shape[0]:
                        model.train(sess, get_feed_dict(args, model, td, start, start + bs))
                        start += bs
                    eval_auc, _ = ctr_eval(args, sess, model, eval_data, bs)
                    if eval_auc > best_auc:
                        best_auc = eval_auc
                        rec = _recommended_items(args, sess, model, user_list,
                                                 train_record, item_set, topk_k, bs)
                        best_cov = catalog_coverage(rec, data['n_item'])
        except (tf.errors.ResourceExhaustedError, tf.errors.InvalidArgumentError) as e:
            # an infeasible/too-large config: penalize instead of crashing the run
            if verbose:
                print('  [eval] penalized config (%s)' % type(e).__name__)
            best_auc, best_cov, n_params = 0.0, 0.0, n_params

        info = {'auc': best_auc, 'coverage': best_cov, 'params': int(n_params),
                'n_active_rel': int(n_active(mask)), 'genome': _summarize(genome)}
        obj = [-best_auc, -best_cov, float(n_params)]
        cache[key] = (obj, info)
        return cache[key]

    return evaluate


def _summarize(genome):
    return {k: (list(int(x) for x in v) if k == 'rmask' else
                (round(float(v), 6) if isinstance(v, float) else v))
            for k, v in genome.items()}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_optimization(args):
    dataset = args.dataset
    seed = getattr(args, 'moo_seed', 555)
    pop_size = getattr(args, 'pop_size', 20)
    n_gen = getattr(args, 'n_gen', 15)
    eval_epochs = getattr(args, 'eval_epochs', 5)
    batch_size = getattr(args, 'batch_size', None) or DATASET_BATCH.get(dataset, 1024)
    out_dir = getattr(args, 'out_dir', '../result')
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print('MOKAR / NSGA-II  dataset=%s  pop=%d  gen=%d  eval_epochs=%d  bs=%d'
          % (dataset, pop_size, n_gen, eval_epochs, batch_size))
    print('=' * 70)

    t0 = time.time()
    data = _load_static_data(dataset, seed)
    print('static data: n_user=%d n_item=%d n_entity=%d n_relation=%d'
          % (data['n_user'], data['n_item'], data['n_entity'], data['n_relation']))

    space = build_space(data['n_relation'])
    evaluate = make_evaluator(data, dataset, batch_size, eval_epochs, seed,
                              verbose=getattr(args, 'verbose', True))

    # Reference point for hypervolume: auc/coverage worst case = 0 (so -obj = 0),
    # params worst case = an upper bound from the largest embedding configuration.
    max_dim = max(c for g in space if g['name'] == 'dim' for c in g['choices'])
    ref_params = (data['n_user'] + data['n_entity'] + data['n_relation']) * max_dim * 1.3 + 1e5
    ref_point = [0.0, 0.0, float(ref_params)]

    history_path = os.path.join(out_dir, 'mokar_%s_history.json' % dataset)

    def _on_gen(gen, pop, archive):
        with open(history_path, 'w') as f:
            json.dump({'dataset': dataset, 'evaluated': len(archive)}, f)

    result = run_nsga2(evaluate, space, pop_size=pop_size, n_gen=n_gen, seed=seed,
                       ref_point=ref_point, verbose=getattr(args, 'verbose', True),
                       on_generation=_on_gen)

    # ---- collect & save results ----
    archive = result['archive']
    objs = np.array([ind['obj'] for ind in archive])
    infos = [ind['info'] for ind in archive]

    pf_idx = pareto_front_indices(objs)
    knee = knee_point_index(objs)
    final_hv = hypervolume(objs, ref_point)

    def _row(i):
        info = infos[i]
        return {'auc': info['auc'], 'coverage': info['coverage'],
                'params': info['params'], 'n_active_rel': info['n_active_rel'],
                'genome': info['genome']}

    out = {
        'dataset': dataset, 'seed': seed,
        'budget': {'pop_size': pop_size, 'n_gen': n_gen, 'eval_epochs': eval_epochs,
                   'evaluations': len(archive)},
        'ref_point': ref_point, 'final_hypervolume': final_hv,
        'history': result['history'],
        'pareto': [_row(int(i)) for i in pf_idx],
        'knee': _row(int(knee)),
        'all': [_row(i) for i in range(len(archive))],
        'elapsed_sec': round(time.time() - t0, 1),
    }
    out_path = os.path.join(out_dir, 'mokar_%s_results.json' % dataset)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    # human-readable Pareto CSV
    csv_path = os.path.join(out_dir, 'mokar_%s_pareto.csv' % dataset)
    with open(csv_path, 'w') as f:
        f.write('auc,coverage,params,n_active_rel,dim,n_layers,n_iter,nss,lr,l2,alpha,aggregator,att\n')
        for i in pf_idx:
            g = infos[i]['genome']
            f.write('%.4f,%.4f,%d,%d,%s,%s,%s,%s,%g,%g,%.3f,%s,%s\n' % (
                infos[i]['auc'], infos[i]['coverage'], infos[i]['params'],
                infos[i]['n_active_rel'], g['dim'], g['n_layers'], g['n_iter'],
                g['nss'], g['lr'], g['l2'], g['alpha'], g['aggregator'], g['att']))

    print('\n' + '=' * 70)
    print('DONE in %.1fs  evaluations=%d  final HV=%.5f' % (out['elapsed_sec'], len(archive), final_hv))
    print('Pareto solutions: %d' % len(pf_idx))
    k = out['knee']
    print('Knee solution: AUC=%.4f  coverage=%.4f  params=%d  active_rel=%d'
          % (k['auc'], k['coverage'], k['params'], k['n_active_rel']))
    print('Saved: %s' % out_path)
    print('Saved: %s' % csv_path)
    print('=' * 70)
    return out


def _build_argparser():
    p = argparse.ArgumentParser(description='MOKAR NSGA-II optimizer for LGKAT')
    p.add_argument('--dataset', type=str, default='music')
    p.add_argument('--pop_size', type=int, default=20)
    p.add_argument('--n_gen', type=int, default=15)
    p.add_argument('--eval_epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=0, help='0 = dataset default')
    p.add_argument('--moo_seed', type=int, default=555)
    p.add_argument('--out_dir', type=str, default='../result')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=1)
    return p


if __name__ == '__main__':
    a = _build_argparser().parse_args()
    if a.batch_size == 0:
        a.batch_size = None
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu_id)
    run_optimization(a)
