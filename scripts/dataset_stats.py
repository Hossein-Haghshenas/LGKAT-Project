"""
Print dataset statistics for every prepared dataset (OL-LGKAT Phase 2).

For each data/<name>/ with ratings_final + kg_final, reports #users, #items,
#interactions (+#positives), #entities, #relations, #triples, sparsity, and whether
interaction timestamps are present (4th ratings column).

Run from repo root or anywhere:
    conda run -n lgkat python scripts/dataset_stats.py
"""
import os
import numpy as np

DATASETS = ['music', 'book', 'restaurant', 'movie', 'ml1m']
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def _load(ds, name):
    base = os.path.join(ROOT, ds, name)
    if os.path.exists(base + '.npy'):
        return np.load(base + '.npy')
    if os.path.exists(base + '.txt'):
        return np.loadtxt(base + '.txt', dtype=np.int64)
    return None


def stats_for(ds):
    r = _load(ds, 'ratings_final')
    if r is None:
        return None
    if r.ndim == 1:
        r = r.reshape(1, -1)
    kg = _load(ds, 'kg_final')
    n_user = int(len(np.unique(r[:, 0])))
    n_item = int(len(np.unique(r[:, 1])))
    n_inter = int(r.shape[0])
    n_pos = int((r[:, 2] == 1).sum())
    n_entity = int(len(np.unique(np.concatenate([kg[:, 0], kg[:, 2]])))) if kg is not None else 0
    n_rel = int(len(np.unique(kg[:, 1]))) if kg is not None else 0
    n_tri = int(kg.shape[0]) if kg is not None else 0
    has_ts = bool(r.shape[1] >= 4)
    sparsity = 1.0 - n_pos / (n_user * n_item) if n_user and n_item else float('nan')
    return dict(dataset=ds, users=n_user, items=n_item, interactions=n_inter, positives=n_pos,
                entities=n_entity, relations=n_rel, triples=n_tri,
                sparsity=round(sparsity, 5), timestamps=has_ts)


if __name__ == '__main__':
    hdr = ['dataset', 'users', 'items', 'interactions', 'positives', 'entities',
           'relations', 'triples', 'sparsity', 'timestamps']
    print('\t'.join(hdr))
    for ds in DATASETS:
        s = stats_for(ds)
        if s is None:
            print('%s\t(not prepared)' % ds)
            continue
        print('\t'.join(str(s[h]) for h in hdr))
