"""
Timestamp-preserving dataset converter to the project's CTR format (OL-LGKAT Phase 2).

Unlike src/preprocess.py (which drops timestamps), this writes a 4-column
ratings_final.txt:

    user_index   item_index   label(0/1)   timestamp

The base loader only reads columns 0-2, so the base model still runs unchanged; the
Phase 3 temporal split reads column 3.  This script does NOT touch the existing
datasets or src/preprocess.py, so base reproduction is unaffected.

Robustness: items are renumbered to exactly those that appear in ratings_final, and the
KG is remapped so those items occupy entity ids 0..M-1.  This guarantees the contiguous
item-id space the LightGCN branch relies on (entity_emb_matrix[:n_item]) and that every
entity has >=1 triple (required by construct_adj), regardless of dataset coverage.

Currently supports:
  ml1m : RippleNet MovieLens-1M.
         ratings.dat               -> 'UserID::MovieID::Rating::Timestamp'
         item_index2entity_id.txt  -> 'movieId \t satori_entity_id'  (rehashed)
         kg.txt                    -> 'head \t relation \t tail'      (rehashed)
                                      (= kg_part1_rehashed + kg_part2_rehashed)

Run from src/:
    python preprocess_temporal.py --dataset ml1m --neg_ratio 1.0 --seed 555
"""

import argparse
import json
import os

import numpy as np

# Per-dataset raw format. cols = (user, item, rating, timestamp) column indices.
CONFIG = {
    'ml1m': {
        'rating_file': 'ratings.dat',
        'sep': '::',
        'cols': (0, 1, 2, 3),
        'threshold': 4.0,   # rating >= threshold => positive interaction
    },
}


def read_item_to_entity(path):
    """movieId(str) -> satori_entity_id(str), preserving file order."""
    movie2satori = {}
    for line in open(path, encoding='utf-8'):
        s = line.strip().split('\t')
        if len(s) < 2:
            s = line.strip().split()
        if len(s) < 2:
            continue
        movie2satori[s[0]] = s[1]
    return movie2satori


def read_positives(rating_path, sep, cols, threshold, movie2satori):
    """Collect positive interactions (rating>=threshold) for KG-linked items.

    Returns user_pos: user_old(str) -> list of (movieId(str), ts(int)).
    """
    uc, ic, rc, tc = cols
    user_pos = {}
    n_lines = n_kept = 0
    for line in open(rating_path, encoding='utf-8'):
        a = line.strip().split(sep)
        if len(a) <= max(cols):
            continue
        n_lines += 1
        movie = a[ic]
        if movie not in movie2satori:          # item not in the KG -> skip
            continue
        if float(a[rc]) < threshold:            # not a positive
            continue
        user_pos.setdefault(a[uc], []).append((movie, int(a[tc])))
        n_kept += 1
    return user_pos, n_lines, n_kept


def convert(dataset, data_dir, neg_ratio, seed):
    cfg = CONFIG[dataset]
    rng = np.random.RandomState(seed)

    movie2satori = read_item_to_entity(os.path.join(data_dir, 'item_index2entity_id.txt'))
    user_pos, n_lines, n_pos_raw = read_positives(
        os.path.join(data_dir, cfg['rating_file']), cfg['sep'], cfg['cols'],
        cfg['threshold'], movie2satori)

    # --- item / user renumbering over things that actually appear as positives ---
    pos_movies = sorted({m for items in user_pos.values() for m, _ in items})
    movie2item = {m: i for i, m in enumerate(pos_movies)}          # movieId -> new item id 0..M-1
    n_item = len(pos_movies)
    users = sorted(user_pos.keys(), key=lambda x: int(x))
    user2new = {u: i for i, u in enumerate(users)}                 # userId -> new user id 0..U-1
    n_user = len(users)

    # --- write 4-column ratings_final.txt (user item label timestamp) ---
    out_ratings = os.path.join(data_dir, 'ratings_final.txt')
    n_pos = n_neg = 0
    ts_min, ts_max = None, None
    all_item_ids = np.arange(n_item)
    with open(out_ratings, 'w', encoding='utf-8') as w:
        for u_old in users:
            u = user2new[u_old]
            items_ts = [(movie2item[m], ts) for m, ts in user_pos[u_old]]
            pos_ids = {it for it, _ in items_ts}
            for it, ts in items_ts:
                w.write('%d\t%d\t1\t%d\n' % (u, it, ts))
                n_pos += 1
                ts_min = ts if ts_min is None else min(ts_min, ts)
                ts_max = ts if ts_max is None else max(ts_max, ts)
            # negatives: sample from the positive-item pool, exclude this user's positives,
            # assign each negative the timestamp of a paired positive (keeps it in-window)
            n_target = int(round(neg_ratio * len(items_ts)))
            if n_target > 0:
                candidates = np.setdiff1d(all_item_ids, np.fromiter(pos_ids, dtype=int),
                                          assume_unique=False)
                if len(candidates) > 0:
                    k = min(n_target, len(candidates))
                    chosen = rng.choice(candidates, size=k, replace=False)
                    ts_list = [ts for _, ts in items_ts]
                    for j, it in enumerate(chosen):
                        w.write('%d\t%d\t0\t%d\n' % (u, int(it), ts_list[j % len(ts_list)]))
                        n_neg += 1

    # --- remap KG: items occupy entity ids 0..M-1, then other entities ---
    entity2idx = {movie2satori[m]: movie2item[m] for m in pos_movies}
    relation2idx = {}
    entity_cnt = n_item
    n_tri = 0
    items_with_triples = set()
    out_kg = os.path.join(data_dir, 'kg_final.txt')
    with open(out_kg, 'w', encoding='utf-8') as w:
        for line in open(os.path.join(data_dir, 'kg.txt'), encoding='utf-8'):
            a = line.strip().split('\t')
            if len(a) != 3:
                continue
            h, r, t = a
            if h not in entity2idx:
                entity2idx[h] = entity_cnt
                entity_cnt += 1
            if t not in entity2idx:
                entity2idx[t] = entity_cnt
                entity_cnt += 1
            if r not in relation2idx:
                relation2idx[r] = len(relation2idx)
            hi, ti = entity2idx[h], entity2idx[t]
            w.write('%d\t%d\t%d\n' % (hi, relation2idx[r], ti))
            n_tri += 1
            if hi < n_item:
                items_with_triples.add(hi)
            if ti < n_item:
                items_with_triples.add(ti)

    # --- new item_index2entity_id.txt (new item id -> satori id) ---
    with open(os.path.join(data_dir, 'item_index2entity_id.txt'), 'w', encoding='utf-8') as w:
        for m in pos_movies:
            w.write('%d\t%s\n' % (movie2item[m], movie2satori[m]))

    # --- integrity check: construct_adj needs every entity (incl. items) to have a triple
    missing = n_item - len(items_with_triples)
    if missing > 0:
        raise RuntimeError(
            '%d item(s) have no KG triple -> construct_adj would KeyError. '
            'Dataset coverage problem.' % missing)

    stats = {
        'dataset': dataset, 'n_user': n_user, 'n_item': n_item,
        'n_interactions': n_pos + n_neg, 'n_positive': n_pos, 'n_negative': n_neg,
        'neg_ratio': neg_ratio, 'n_entity': entity_cnt, 'n_relation': len(relation2idx),
        'n_triples': n_tri, 'rating_lines_read': n_lines, 'positives_kept_raw': n_pos_raw,
        'sparsity': 1.0 - (n_pos / (n_user * n_item)) if n_user and n_item else None,
        'timestamp_min': ts_min, 'timestamp_max': ts_max,
        'has_timestamps': True, 'seed': seed,
    }
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='ml1m', choices=list(CONFIG.keys()))
    p.add_argument('--data_root', type=str, default='../data')
    p.add_argument('--neg_ratio', type=float, default=1.0, help='negatives per positive')
    p.add_argument('--seed', type=int, default=555)
    args = p.parse_args()

    data_dir = os.path.join(args.data_root, args.dataset)
    print('converting %s (neg_ratio=%g, seed=%d) ...' % (args.dataset, args.neg_ratio, args.seed))
    stats = convert(args.dataset, data_dir, args.neg_ratio, args.seed)

    # remove any stale .npy caches so the loader re-reads the new .txt files
    for f in ('ratings_final.npy', 'kg_final.npy'):
        fp = os.path.join(data_dir, f)
        if os.path.exists(fp):
            os.remove(fp)

    with open(os.path.join(data_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print('wrote ratings_final.txt (4-col incl. timestamp), kg_final.txt, '
          'item_index2entity_id.txt, stats.json under %s' % data_dir)


if __name__ == '__main__':
    main()
