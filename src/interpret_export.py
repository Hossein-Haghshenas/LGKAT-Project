"""
Interpretability export for the Forgetting-Aware Fusion Gate (OL-LGKAT Phase 6).

For a sample user and a set of items, export per item:
  * the fusion gate value g in [0,1] — how much the model trusts the CF (LightGCN) item branch
    vs. (1-g) on the knowledge-graph branch,
  * the item's stability features (age, freq, recency, low-drift),
  * the prediction score.

This is the data behind a case study of *why* the gate fuses the way it does (e.g. stable items
-> higher g / lean CF; cold items -> lean KG prior). TF-dependent.
"""

import json

import numpy as np


def export_gate_case_study(args, sess, model, user, items, tracker=None, window=None, out_path=None):
    bs = args.batch_size
    items = [int(x) for x in list(items)[:bs]]
    if not items:
        return None
    padded = (items + [items[-1]] * (bs - len(items)))[:bs]
    nd = len(eval(args.layer_size))
    feed = {model.user_indices: [int(user)] * bs, model.item_indices: padded,
            model.node_dropout: [0.] * nd, model.mess_dropout: [0.] * nd}

    fetches = {'score': model.scores_normalized}
    if getattr(model, 'fusion_gate', None) is not None:
        fetches['gate'] = model.fusion_gate
    out = sess.run(fetches, feed)
    score = np.reshape(out['score'], (bs,))
    gate = np.reshape(out['gate'], (bs,)) if 'gate' in out else np.full(bs, float('nan'))

    stab = tracker.features(window) if (tracker is not None and window is not None) else None
    recs = []
    for i, it in enumerate(items):
        rec = {'item': it, 'gate_g': float(gate[i]), 'score': float(score[i])}
        if stab is not None:
            rec['stability'] = [float(x) for x in stab[it]]
        recs.append(rec)

    payload = {
        'dataset': args.dataset, 'fusion': getattr(args, 'fusion', 'fixed'),
        'user': int(user), 'window': window,
        'gate_note': 'g in [0,1] = weight on the CF (LightGCN) item branch; (1-g) on the KG branch',
        'stability_note': 'features = [age, freq, recency, low-drift], each in [0,1]; 0 for cold items',
        'items': recs,
    }
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
    return payload
