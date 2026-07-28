# compare legacy vs --fast_eval scores on the same TF weights (music smoke).
# run from repo root:  python scripts/test_fast_eval_parity.py
# or:  cd src && python ../scripts/test_fast_eval_parity.py

import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.chdir(SRC)

import tensorflow as tf
from data_loader import load_data
from model import KGCN
from train_n_runs import get_feed_dict, _score_user_items, topk_settings


def build_args():
    # minimal namespace matching cmd.sh music paper flags (short train)
    class A:
        pass
    args = A()
    args.dataset = 'music'
    args.aggregator = 'concat'
    args.n_epochs = 1
    args.neighbor_sample_size = 8
    args.dim = 32
    args.n_iter = 1
    args.batch_size = 128
    args.l2_weight = 1e-4
    args.ls_weight = 0
    args.lr = 0.005
    args.ratio = 1
    args.layer_size = '[32]'
    args.adj_type = 'norm'
    args.alg_type = 'lightgcn'
    args.model_type = 'KGCN_LightGCN'
    args.node_dropout = '[0.1]'
    args.mess_dropout = '[0.1]'
    args.node_dropout_flag = 1
    args.alpha = 0.5
    args.smoothing_steps = 8
    args.pretrain = 0
    args.att = 'product'
    args.agg_type = 'weighted_avg'
    args.runs = 1
    args.gpu_id = 0
    args.seed = 555
    args.use_cl = 0
    args.cl_lambda = 0.1
    args.cl_tau = 0.2
    args.fusion = 'fixed'
    args.fast_eval = 1
    args.topk_user_num = 20
    args.topk_k_list = '[2,10,50,100]'
    args.report_ndcg = 1
    return args


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = build_args()
    data = load_data(args)
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    user_item_adj = data[9]

    tf.compat.v1.reset_default_graph()
    model = KGCN(args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, user_item_adj)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # a few train steps so weights leave init
        start = 0
        steps = 0
        while start + args.batch_size <= train_data.shape[0] and steps < 20:
            model.train(sess, get_feed_dict(args, model, train_data, start, start + args.batch_size))
            start += args.batch_size
            steps += 1

        cache = model.cache_cf_embeddings(sess)

        # CTR batch parity
        feed = get_feed_dict(args, model, eval_data, 0, args.batch_size, flag=False)
        _, slow = model.get_scores(sess, feed)
        _, fast = model.get_scores_cached(sess, feed, cache)
        diff = np.max(np.abs(np.asarray(slow) - np.asarray(fast)))
        print('ctr_batch max|slow-fast|=%.6e  mean=%.6e' % (
            diff, np.mean(np.abs(np.asarray(slow) - np.asarray(fast)))))

        # top-K timing + ranking agreement on a few users
        user_list, train_record, test_record, item_set, k_list = topk_settings(
            True, train_data, eval_data, n_item, seed=args.seed, args=args)
        users = list(user_list[:5])

        t0 = time.time()
        slow_ranks = []
        for u in users:
            slow_ranks.append(_score_user_items(
                args, sess, model, u, train_record, item_set, args.batch_size, None))
        t_slow = time.time() - t0

        t0 = time.time()
        fast_ranks = []
        for u in users:
            fast_ranks.append(_score_user_items(
                args, sess, model, u, train_record, item_set, args.batch_size, cache))
        t_fast = time.time() - t0

        top10_match = []
        for s, f in zip(slow_ranks, fast_ranks):
            top10_match.append(s[:10] == f[:10])
        print('topk users=%d  slow=%.2fs  fast=%.2fs  speedup=%.2fx' % (
            len(users), t_slow, t_fast, (t_slow / t_fast) if t_fast > 0 else float('inf')))
        print('top10_rank_identical: %s (%d/%d)' % (
            all(top10_match), sum(top10_match), len(top10_match)))

        ok = diff < 1e-4 and all(top10_match)
        print('RESULT', 'PASS' if ok else 'FAIL')
        return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
