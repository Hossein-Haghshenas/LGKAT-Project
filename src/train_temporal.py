"""
Temporal / streaming protocol (OL-LGKAT Phase 3) + Forgetting-Aware Fusion Gate incremental
loop with checkpoint/resume (OL-LGKAT Phase 4).

Split the (timestamped) interactions into T contiguous time windows; for t = 0..T-2, train on
window t (or the cumulative history) and evaluate on the *next* window t+1.

Modes:
  batch        : reference / upper bound. Each step trains a FRESH model on the cumulative
                 history (windows 0..t).
  incremental  : the online core. Maintain ONE model; warm-start each step from the previous
                 step's checkpoint (weights + Adam slots + stability table) and train only on
                 window t.

The UI adjacency for LightGCN is rebuilt each step from the cumulative interactions (<= t)
only, so it is causal (no future edges -> no leakage).

Phase 4 additions:
  * forgetting_gate fusion: a per-item StabilityTracker (src/stability.py) is fed into the gate
    each window (stability as of ENTERING the window) and updated after training.
  * checkpoint/resume after EVERY window: TF checkpoint (weights+optimizer+stability variable)
    + meta (next window index) + stability tracker (.npz) + accumulated results. Re-run with
    --resume 1 to continue; --stop_after_window K stops early (for stop/resume tests).
  * deterministic per-window seed (seed + t) so a stopped-and-resumed run reproduces a straight
    run window-for-window; --deterministic 1 also pins single-thread + TF_DETERMINISTIC_OPS.

Reuses the base model (model.KGCN) and the shared eval helpers from train_n_runs.
Invoked from main.py when --split temporal.
"""

import json
import os
import shutil
import time

import numpy as np
import tensorflow as tf

from model import KGCN
from data_loader import temporal_windows, build_user_item_adj
from train_n_runs import get_feed_dict, ctr_eval, topk_eval, topk_settings
from stability import StabilityTracker


def _at(vec, k_list, k):
    return float(vec[k_list.index(k)]) if k in k_list else float('nan')


def _state_dir(args):
    return '../Pretrain/temporal/{}_{}_{}_{}/'.format(
        args.model_type, args.dataset, args.protocol, getattr(args, 'fusion', 'fixed'))


def _result_tag(args):
    parts = [args.model_type, args.dataset, args.protocol, getattr(args, 'fusion', 'fixed')]
    bl = getattr(args, 'baseline', 'none')
    if bl and bl != 'none':
        parts.append(bl)
    return '_'.join(parts)


def _estimate_fisher(args, model, sess, data, max_batches=5):
    """Empirical diagonal Fisher = mean over a few batches of (d base_loss / d theta)^2."""
    acc = None
    n = 0
    s = 0
    while s + args.batch_size <= data.shape[0] and n < max_batches:
        f = model.compute_fisher(sess, get_feed_dict(args, model, data, s, s + args.batch_size, flag=False))
        acc = f if acc is None else {k: acc[k] + f[k] for k in acc}
        n += 1
        s += args.batch_size
    if acc is None:
        return None
    return {k: acc[k] / n for k in acc}


def run_temporal(args, data):
    # honor --gpu_id before any TF session is created (e.g. --gpu_id -1 => CPU), so the temporal
    # path does not depend on a shell CUDA_VISIBLE_DEVICES.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    (n_user, n_item, n_entity, n_relation, rating_np, adj_entity, adj_relation) = data
    T = int(args.n_windows)
    mode = args.protocol
    fusion = getattr(args, 'fusion', 'fixed')
    forgetting = (fusion == 'forgetting_gate')
    deterministic = bool(int(getattr(args, 'deterministic', 0)))
    resume = bool(int(getattr(args, 'resume', 0)))
    stop_after = int(getattr(args, 'stop_after_window', -1))
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    windows = temporal_windows(rating_np, T)
    print('temporal protocol: mode=%s fusion=%s windows=%d (sizes=%s) deterministic=%d'
          % (mode, fusion, T, [int(w.shape[0]) for w in windows], int(deterministic)))

    state_dir = _state_dir(args)
    ckpt_path = os.path.join(state_dir, 'model.ckpt')
    meta_path = os.path.join(state_dir, 'meta.json')
    results_path = os.path.join(state_dir, 'results.json')
    track_path = os.path.join(state_dir, 'stability.npz')

    # a fresh run clears stale resume state; --resume keeps it
    if not resume and os.path.isdir(state_dir):
        shutil.rmtree(state_dir)
    os.makedirs(state_dir, exist_ok=True)

    tracker = StabilityTracker(n_item, int(getattr(args, 'n_stability_feats', 4))) if forgetting else None
    start = 0
    results = []
    if resume and os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding='utf-8'))
        start = int(meta['next_window'])
        if os.path.exists(results_path):
            results = json.load(open(results_path, encoding='utf-8'))
        if forgetting and os.path.exists(track_path):
            tracker = StabilityTracker.load_npz(track_path)
        print('[resume] continuing temporal run from window index %d' % start)

    k_list_ref = results[-1]['k_list'] if results else None
    ewc = (getattr(args, 'baseline', 'none') == 'ewc' and float(getattr(args, 'ewc_lambda', 0.0)) > 0)
    ewc_F = None
    ewc_star = None
    for t in range(start, T - 1):
        t_start = time.time()
        cumulative = np.concatenate(windows[:t + 1], axis=0)   # history <= t (causal)
        eval_win = windows[t + 1]
        train_step = cumulative if mode == 'batch' else windows[t]
        ui_adj = build_user_item_adj(args, n_user, n_item, cumulative)

        # deterministic per-window seed: window t's randomness depends only on (seed, t), so a
        # stopped-and-resumed run reproduces a straight run window-for-window.
        seed_t = int(args.seed) + t
        np.random.seed(seed_t)
        # Phase 5 experience-replay baseline: augment window t with a causal random sample of
        # past interactions (windows < t). Sampling is seeded per window so resume stays exact.
        if getattr(args, 'baseline', 'none') == 'replay' and mode == 'incremental' and t > 0:
            past = np.concatenate(windows[:t], axis=0)
            k = min(past.shape[0], int(float(getattr(args, 'replay_ratio', 1.0)) * windows[t].shape[0]))
            if k > 0:
                idx = np.random.choice(past.shape[0], size=k, replace=False)
                train_step = np.concatenate([windows[t], past[idx]], axis=0)
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(seed_t)
        model = KGCN(args, n_user, n_item, n_entity, n_relation,
                     adj_entity, adj_relation, ui_adj, pretrain=None)
        saver = tf.compat.v1.train.Saver()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        if deterministic:
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads = 1
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            # incremental warm-start from the previous window (weights + Adam slots + stability var)
            if mode == 'incremental' and os.path.exists(ckpt_path + '.index'):
                saver.restore(sess, ckpt_path)
            # set the per-item stability table as of ENTERING window t (history < t)
            if forgetting:
                model.update_stability(sess, tracker.features(t))
            if ewc and ewc_F is not None:
                model.set_ewc(sess, ewc_F, ewc_star)   # activate EWC anchor for this window

            for _ in range(args.n_epochs):
                np.random.shuffle(train_step)
                s = 0
                while s + args.batch_size <= train_step.shape[0]:
                    model.train(sess, get_feed_dict(args, model, train_step, s, s + args.batch_size))
                    s += args.batch_size

            eval_auc, eval_f1 = ctr_eval(args, sess, model, eval_win, args.batch_size)
            user_list, train_rec, test_rec, item_set, k_list = topk_settings(
                True, cumulative, eval_win, n_item, seed=args.seed, args=args)
            prec, rec, ndcg = topk_eval(args, sess, model, user_list, train_rec, test_rec,
                                        item_set, k_list, args.batch_size)

            # fold window t's items + post-training item embeddings into the stability stats
            if forgetting:
                _, entity_emb, _ = model.get_embeddings(sess)
                tracker.update(t, windows[t][:, 1], entity_emb[:n_item])

            # EWC: accumulate diagonal Fisher + re-anchor params at this window's solution
            if ewc:
                F_new = _estimate_fisher(args, model, sess, train_step)
                if F_new is not None:
                    ewc_F = F_new if ewc_F is None else {k: ewc_F[k] + F_new[k] for k in ewc_F}
                    ewc_star = model.get_param_values(sess)

            # Phase 6: cold-start buckets + gate interpretability at the final evaluated window
            if t == T - 2:
                if int(getattr(args, 'coldstart', 0)):
                    from coldstart_eval import bucketed_topk_eval
                    boundaries = list(eval(getattr(args, 'coldstart_buckets', '[5,20,50]')))
                    cb = bucketed_topk_eval(args, sess, model, user_list, train_rec, test_rec,
                                            item_set, k_list, args.batch_size, boundaries)
                    cs_path = os.path.join('../result/temporal', _result_tag(args) + '_coldstart.json')
                    json.dump({'dataset': args.dataset, 'fusion': fusion, 'window': t + 1,
                               'k_list': list(k_list), 'buckets': cb},
                              open(cs_path, 'w', encoding='utf-8'), indent=2)
                    if 10 in k_list:
                        print('[coldstart] per-bucket NDCG@10:',
                              {b: round(cb[b]['ndcg'][10], 4) for b in cb})
                if int(getattr(args, 'interpret', 0)) and getattr(model, 'fusion_gate', None) is not None:
                    from coldstart_eval import _rank_user
                    from interpret_export import export_gate_case_study
                    iu = max(user_list, key=lambda u: len(test_rec[u]))
                    ranked = _rank_user(args, sess, model, iu, train_rec[iu], item_set, args.batch_size)
                    ip_path = os.path.join('../result/temporal', _result_tag(args) + '_interpret.json')
                    export_gate_case_study(args, sess, model, iu,
                                           ranked[:int(getattr(args, 'interpret_topk', 10))],
                                           tracker=tracker, window=t, out_path=ip_path)
                    print('[interpret] gate case study for user %d -> %s' % (iu, ip_path))

            if mode == 'incremental':
                saver.save(sess, ckpt_path)

        k_list_ref = k_list
        elapsed = time.time() - t_start
        row = {'window': t + 1, 'mode': mode, 'fusion': fusion, 'seconds': round(elapsed, 2),
               'train_rows': int(train_step.shape[0]), 'eval_rows': int(eval_win.shape[0]),
               'n_eval_users': int(len(user_list)),
               'auc': float(eval_auc), 'f1': float(eval_f1),
               'precision': [float(x) for x in prec], 'recall': [float(x) for x in rec],
               'ndcg': [float(x) for x in ndcg], 'k_list': list(k_list)}
        results.append(row)

        # persist resume state after every window
        json.dump({'next_window': t + 1, 'protocol': mode, 'fusion': fusion},
                  open(meta_path, 'w', encoding='utf-8'))
        json.dump(results, open(results_path, 'w', encoding='utf-8'))
        if forgetting:
            tracker.save_npz(track_path)

        print('[%s/%s] eval window %d  train_rows=%d  %.1fs  AUC=%.4f F1=%.4f R@10=%.4f NDCG@10=%.4f'
              % (mode, fusion, t + 1, train_step.shape[0], elapsed, eval_auc, eval_f1,
                 _at(rec, k_list, 10), _at(ndcg, k_list, 10)))

        if 0 <= stop_after <= t:
            print('[stop] stopping after window index %d (--stop_after_window)' % t)
            break

    return results, k_list_ref


def run_temporal_and_log(args, data):
    results, k_list = run_temporal(args, data)
    out_dir = '../result/temporal'
    os.makedirs(out_dir, exist_ok=True)
    tag = _result_tag(args)

    # JSON (full detail)
    with open(os.path.join(out_dir, tag + '.json'), 'w', encoding='utf-8') as f:
        json.dump({'dataset': args.dataset, 'model_type': args.model_type,
                   'protocol': args.protocol, 'fusion': getattr(args, 'fusion', 'fixed'),
                   'baseline': getattr(args, 'baseline', 'none'),
                   'replay_ratio': float(getattr(args, 'replay_ratio', 1.0)),
                   'ewc_lambda': float(getattr(args, 'ewc_lambda', 0.0)),
                   'n_windows': int(args.n_windows), 'n_epochs': int(args.n_epochs),
                   'seed': int(args.seed), 'k_list': k_list, 'windows': results}, f, indent=2)

    # CSV (one row per evaluated window) — easy to plot
    csv_path = os.path.join(out_dir, tag + '.csv')
    ks = k_list or []
    header = ['window', 'mode', 'fusion', 'train_rows', 'eval_rows', 'n_eval_users', 'auc', 'f1']
    for k in ks:
        header += ['P@%d' % k, 'R@%d' % k, 'N@%d' % k]
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for r in results:
            cells = [r['window'], r['mode'], r.get('fusion', getattr(args, 'fusion', 'fixed')),
                     r['train_rows'], r['eval_rows'], r['n_eval_users'],
                     '%.6f' % r['auc'], '%.6f' % r['f1']]
            for j in range(len(ks)):
                cells += ['%.6f' % r['precision'][j], '%.6f' % r['recall'][j],
                          '%.6f' % r['ndcg'][j]]
            f.write(','.join(str(c) for c in cells) + '\n')

    print('temporal results written:\n  %s\n  %s' % (csv_path, os.path.join(out_dir, tag + '.json')))
    return results
