"""
Cold-start / sparsity bucketed evaluation (OL-LGKAT Phase 6).

Buckets the evaluated users by how many interactions they have in the training history and
reports Recall@K / NDCG@K per bucket, so we can see how each method does on cold (few-history)
vs. warm (rich-history) users. Reuses the same ranking procedure as train_n_runs.topk_eval; only
the aggregation is grouped by bucket.

TF-dependent (calls model.get_scores), so kept out of the TF-free evaluation.py.
"""

import numpy as np

from evaluation import ndcg_at_k, recall_at_k, precision_at_k


def bucket_of(n, boundaries):
    """Map an interaction count n to a bucket label given ascending boundaries, e.g.
    boundaries=[5,20,50] -> '1-5', '6-20', '21-50', '51+'."""
    prev = 0
    for b in boundaries:
        if n <= b:
            return '%d-%d' % (prev + 1, b)
        prev = b
    return '%d+' % (prev + 1)


def _rank_user(args, sess, model, user, train_items, item_set, batch_size):
    test_item_list = list(item_set - train_items)
    nd = len(eval(args.layer_size))
    item_score_map = {}
    start = 0
    while start + batch_size <= len(test_item_list):
        items, scores = model.get_scores(sess, {
            model.user_indices: [user] * batch_size,
            model.item_indices: test_item_list[start:start + batch_size],
            model.node_dropout: [0.] * nd, model.mess_dropout: [0.] * nd})
        for it, sc in zip(items, scores):
            item_score_map[it] = sc
        start += batch_size
    if start < len(test_item_list):
        pad = test_item_list[start:] + [test_item_list[-1]] * (batch_size - len(test_item_list) + start)
        items, scores = model.get_scores(sess, {
            model.user_indices: [user] * batch_size, model.item_indices: pad,
            model.node_dropout: [0.] * nd, model.mess_dropout: [0.] * nd})
        for it, sc in zip(items, scores):
            item_score_map[it] = sc
    return [i[0] for i in sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)]


def bucketed_topk_eval(args, sess, model, user_list, train_record, test_record,
                       item_set, k_list, batch_size, boundaries):
    """Return {bucket_label: {'count': n, 'recall': {k: v}, 'ndcg': {k: v}, 'precision': {k: v}}}."""
    acc = {}  # bucket -> {'recall':{k:[...]}, 'ndcg':{...}, 'precision':{...}, 'count':n}
    for user in user_list:
        bname = bucket_of(len(train_record[user]), boundaries)
        ranked = _rank_user(args, sess, model, user, train_record[user], item_set, batch_size)
        rel = test_record[user]
        b = acc.setdefault(bname, {'recall': {k: [] for k in k_list},
                                   'ndcg': {k: [] for k in k_list},
                                   'precision': {k: [] for k in k_list}, 'count': 0})
        for k in k_list:
            b['recall'][k].append(recall_at_k(ranked, rel, k))
            b['ndcg'][k].append(ndcg_at_k(ranked, rel, k))
            b['precision'][k].append(precision_at_k(ranked, rel, k))
        b['count'] += 1

    out = {}
    for bname, b in acc.items():
        out[bname] = {
            'count': b['count'],
            'recall': {k: float(np.mean(b['recall'][k])) for k in k_list},
            'ndcg': {k: float(np.mean(b['ndcg'][k])) for k in k_list},
            'precision': {k: float(np.mean(b['precision'][k])) for k in k_list},
        }
    return out
