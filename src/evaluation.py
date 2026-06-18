"""
Single reusable evaluation module for OL-LGKAT (Phase 1).

Pure, TensorFlow-free ranking metrics (so they are trivially unit-testable on CPU)
plus a structured results writer (CSV/JSON).

Ranking metrics use binary relevance: an item is relevant to a user iff it is in the
user's held-out positive set. NDCG uses the standard log2 position discount with an
ideal DCG that places min(|relevant|, k) relevant items at the top.

These functions reproduce the exact Precision@K / Recall@K already used by the base
model, and add NDCG@K alongside them. Keeping them here means train_n_runs.py and the
unit tests share one implementation.
"""

import os
import csv
import json
import math
from datetime import datetime

import numpy as np


# --------------------------------------------------------------------------------------
# Ranking metrics (binary relevance)
# --------------------------------------------------------------------------------------
def _dcg(gains):
    """Discounted cumulative gain for gains given in rank order (rank 1 first)."""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def precision_at_k(ranked_items, relevant_set, k):
    """|top-k ∩ relevant| / k  (matches the base model's definition)."""
    if k <= 0:
        return 0.0
    hit = len(set(ranked_items[:k]) & relevant_set)
    return hit / k


def recall_at_k(ranked_items, relevant_set, k):
    """|top-k ∩ relevant| / |relevant|  (matches the base model's definition)."""
    if len(relevant_set) == 0:
        return 0.0
    hit = len(set(ranked_items[:k]) & relevant_set)
    return hit / len(relevant_set)


def ndcg_at_k(ranked_items, relevant_set, k):
    """
    Normalized DCG@k with binary relevance.

        DCG@k  = sum_{i=1..k} rel_i / log2(i + 1)
        IDCG@k = sum_{i=1..min(|rel|,k)} 1 / log2(i + 1)
        NDCG@k = DCG@k / IDCG@k

    Returns 0.0 when there are no relevant items or k <= 0.
    """
    if k <= 0 or len(relevant_set) == 0:
        return 0.0
    gains = [1.0 if item in relevant_set else 0.0 for item in ranked_items[:k]]
    dcg = _dcg(gains)
    ideal_hits = min(len(relevant_set), k)
    idcg = _dcg([1.0] * ideal_hits)
    return float(dcg / idcg) if idcg > 0 else 0.0


def ranking_metrics_at_ks(ranked_items, relevant_set, k_list):
    """Compute precision/recall/ndcg at every k in k_list for one user.

    Returns dict of lists aligned with k_list: {'precision': [...], 'recall': [...], 'ndcg': [...]}.
    """
    relevant_set = set(relevant_set)
    return {
        'precision': [precision_at_k(ranked_items, relevant_set, k) for k in k_list],
        'recall': [recall_at_k(ranked_items, relevant_set, k) for k in k_list],
        'ndcg': [ndcg_at_k(ranked_items, relevant_set, k) for k in k_list],
    }


def ctr_metrics(labels, scores, threshold=0.5):
    """AUC and F1 for CTR-style binary prediction. (sklearn imported lazily.)"""
    from sklearn.metrics import f1_score, roc_auc_score
    auc = roc_auc_score(y_true=labels, y_score=scores)
    preds = (np.asarray(scores) >= threshold).astype(int)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return float(auc), float(f1)


# --------------------------------------------------------------------------------------
# Structured results writer (CSV + JSON), additive — never touches legacy result files
# --------------------------------------------------------------------------------------
PAPER_KS = [2, 10, 50, 100]  # stable CSV schema regardless of --topk_k_list


def _mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(np.mean(arr)), float(np.std(arr))


def write_structured_results(args, k_list, auc_arr, f1_arr,
                             precision_runs, recall_runs, ndcg_runs=None,
                             out_dir='../result/structured'):
    """
    Persist one experiment's aggregated metrics as both JSON (full detail) and a row in
    a shared runs.csv (stable paper-K schema). Returns (json_path, csv_path).

    *_runs are arrays shaped [n_runs, len(k_list)] of best-epoch test metrics per run.
    ndcg_runs may be None when --report_ndcg 0.
    """
    os.makedirs(out_dir, exist_ok=True)
    precision_runs = np.asarray(precision_runs, dtype=float)
    recall_runs = np.asarray(recall_runs, dtype=float)
    has_ndcg = ndcg_runs is not None
    if has_ndcg:
        ndcg_runs = np.asarray(ndcg_runs, dtype=float)

    auc_mean, auc_std = _mean_std(auc_arr)
    f1_mean, f1_std = _mean_std(f1_arr)

    per_k = {}
    for j, k in enumerate(k_list):
        entry = {
            'precision': dict(zip(('mean', 'std'), _mean_std(precision_runs[:, j]))),
            'recall': dict(zip(('mean', 'std'), _mean_std(recall_runs[:, j]))),
        }
        entry['precision']['per_run'] = precision_runs[:, j].tolist()
        entry['recall']['per_run'] = recall_runs[:, j].tolist()
        if has_ndcg:
            nm, ns = _mean_std(ndcg_runs[:, j])
            entry['ndcg'] = {'mean': nm, 'std': ns, 'per_run': ndcg_runs[:, j].tolist()}
        per_k['@{}'.format(k)] = entry

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    payload = {
        'timestamp': timestamp,
        'dataset': args.dataset,
        'model_type': args.model_type,
        'att': args.att,
        'seed': getattr(args, 'seed', None),
        'config': {k: _jsonify(v) for k, v in vars(args).items()},
        'k_list': list(k_list),
        'metrics': {
            'auc': {'mean': auc_mean, 'std': auc_std, 'per_run': list(map(float, auc_arr))},
            'f1': {'mean': f1_mean, 'std': f1_std, 'per_run': list(map(float, f1_arr))},
            'per_k': per_k,
        },
    }

    tag = _att_tag(args)
    json_path = os.path.join(
        out_dir, '{}_{}_{}_{}.json'.format(args.model_type, args.dataset, tag, timestamp))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(out_dir, 'runs.csv')
    _append_csv_row(csv_path, args, tag, timestamp, k_list,
                    auc_mean, auc_std, f1_mean, f1_std,
                    precision_runs, recall_runs, ndcg_runs, has_ndcg)
    return json_path, csv_path


def _append_csv_row(csv_path, args, tag, timestamp, k_list,
                    auc_mean, auc_std, f1_mean, f1_std,
                    precision_runs, recall_runs, ndcg_runs, has_ndcg):
    k_index = {k: j for j, k in enumerate(k_list)}

    def cell(runs_arr, k):
        if k in k_index and runs_arr is not None:
            m, s = _mean_std(runs_arr[:, k_index[k]])
            return m, s
        return float('nan'), float('nan')

    header = ['timestamp', 'dataset', 'model_type', 'att', 'fusion', 'alpha', 'dim', 'lr',
              'n_epochs', 'neighbor_sample_size', 'n_iter', 'batch_size', 'runs',
              'auc_mean', 'auc_std', 'f1_mean', 'f1_std']
    for k in PAPER_KS:
        header += ['P@{}_mean'.format(k), 'R@{}_mean'.format(k), 'N@{}_mean'.format(k)]

    row = [timestamp, args.dataset, args.model_type, tag,
           getattr(args, 'fusion', ''), getattr(args, 'alpha', ''), args.dim, args.lr,
           args.n_epochs, args.neighbor_sample_size, args.n_iter, args.batch_size,
           args.runs, _r(auc_mean), _r(auc_std), _r(f1_mean), _r(f1_std)]
    for k in PAPER_KS:
        p, _ = cell(precision_runs, k)
        r, _ = cell(recall_runs, k)
        n, _ = cell(ndcg_runs if has_ndcg else None, k)
        row += [_r(p), _r(r), _r(n)]

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def _att_tag(args):
    """Mirror train_n_runs file tagging so structured files line up with legacy ones."""
    return getattr(args, 'att', 'att')


def _jsonify(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _r(x, nd=6):
    try:
        if x != x:  # NaN
            return ''
        return round(float(x), nd)
    except (TypeError, ValueError):
        return ''
