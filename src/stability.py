"""
Per-item historical stability tracker for the Forgetting-Aware Fusion Gate (OL-LGKAT Phase 4).

Across temporal windows it maintains, for each item, four bounded [0,1] signals that describe
how "established / stable" the item is. These are fed into the gate so it can lean on the
collaborative (CF) branch for stable items (preserve learned signal -> resist forgetting) and
on the knowledge-graph prior for new / volatile items (good cold-start):

    age      = (t - first_seen) / t        older item            -> 1
    freq     = #windows_seen / t           frequently active     -> 1
    recency  = 1 - (t - last_seen) / t      seen recently          -> 1
    stab     = 1 / (1 + drift_ema)          low embedding drift   -> 1

Items never seen yet get an all-zero feature vector, so with the gate's identity init
(b = logit(alpha)) a brand-new item starts at the proven fixed-alpha fusion.

Pure NumPy (no TensorFlow) so it is unit-testable, and it serializes to / from a plain dict
for checkpoint/resume.
"""

import numpy as np


class StabilityTracker:
    def __init__(self, n_item, n_stab=4, ema=0.5):
        self.n_item = int(n_item)
        self.n_stab = int(n_stab)
        self.ema = float(ema)
        self.first_seen = np.full(self.n_item, -1, dtype=np.int64)
        self.last_seen = np.full(self.n_item, -1, dtype=np.int64)
        self.count = np.zeros(self.n_item, dtype=np.int64)      # #windows the item appeared in
        self.drift = np.zeros(self.n_item, dtype=np.float32)    # EMA of ||emb_t - emb_{t-1}||
        self.prev_emb = None                                    # item embeddings at previous update

    def features(self, t):
        """Stability features as of ENTERING window t (history over windows < t).

        Returns float32 array [n_item, n_stab] in [0, 1]. At t == 0 everything is 0.
        """
        denom = float(max(1, t))
        seen = self.count > 0
        age = np.where(self.first_seen >= 0, (t - self.first_seen) / denom, 0.0)
        freq = self.count / denom
        recency = np.where(self.last_seen >= 0, 1.0 - (t - self.last_seen) / denom, 0.0)
        stab = np.where(seen, 1.0 / (1.0 + self.drift), 0.0)
        feats = np.stack([age, freq, recency, stab], axis=1).astype(np.float32)
        return np.clip(feats, 0.0, 1.0)[:, : self.n_stab]

    def update(self, t, window_item_ids, item_emb=None):
        """Fold window t's observed items (and optionally new item embeddings) into the stats."""
        seen = np.unique(np.asarray(window_item_ids, dtype=np.int64))
        seen = seen[(seen >= 0) & (seen < self.n_item)]
        fresh = seen[self.first_seen[seen] < 0]
        self.first_seen[fresh] = t
        self.last_seen[seen] = t
        self.count[seen] += 1
        if item_emb is not None:
            item_emb = np.asarray(item_emb, dtype=np.float32)
            if self.prev_emb is not None and self.prev_emb.shape == item_emb.shape:
                d = np.linalg.norm(item_emb - self.prev_emb, axis=1)
                self.drift = (1.0 - self.ema) * self.drift + self.ema * d.astype(np.float32)
            self.prev_emb = item_emb.copy()

    # --- checkpoint / resume support ---
    def state_dict(self):
        return {
            'n_item': self.n_item, 'n_stab': self.n_stab, 'ema': self.ema,
            'first_seen': self.first_seen, 'last_seen': self.last_seen,
            'count': self.count, 'drift': self.drift,
            'prev_emb': self.prev_emb if self.prev_emb is not None else np.empty(0, dtype=np.float32),
            'has_prev': self.prev_emb is not None,
        }

    def load_state(self, d):
        self.n_item = int(d['n_item']); self.n_stab = int(d['n_stab']); self.ema = float(d['ema'])
        self.first_seen = np.asarray(d['first_seen'], dtype=np.int64)
        self.last_seen = np.asarray(d['last_seen'], dtype=np.int64)
        self.count = np.asarray(d['count'], dtype=np.int64)
        self.drift = np.asarray(d['drift'], dtype=np.float32)
        self.prev_emb = np.asarray(d['prev_emb'], dtype=np.float32) if bool(d['has_prev']) else None
        return self

    def save_npz(self, path):
        np.savez(path, **self.state_dict())

    @classmethod
    def load_npz(cls, path):
        d = np.load(path, allow_pickle=False)
        obj = cls(int(d['n_item']), int(d['n_stab']), float(d['ema']))
        return obj.load_state(d)
