"""
objectives.py
=============
Objective helpers for MOKAR multi-objective optimization.

Three objectives are used (the optimizer MINIMIZES, so accuracy/diversity are
negated by the caller):

  * accuracy   -> validation AUC (maximize)
  * diversity  -> catalog coverage@K of the recommendations (maximize)
  * efficiency -> number of trainable parameters (minimize)

This module deliberately contains only pure / TF-agnostic helpers (plus one
small helper that reads trainable-variable shapes from a session) so it can be
unit-tested and reused.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Diversity
# --------------------------------------------------------------------------- #
def catalog_coverage(recommended_items_per_user, n_item):
    """Catalog coverage@K.

    Args:
        recommended_items_per_user: iterable of iterables; the top-K item ids
            recommended for each evaluated user.
        n_item: total number of items in the catalog.

    Returns:
        Fraction of the catalog that appears in at least one user's top-K list,
        in [0, 1]. Higher = more diverse / less popularity-biased.
    """
    if n_item <= 0:
        return 0.0
    recommended = set()
    for items in recommended_items_per_user:
        recommended.update(int(i) for i in items)
    return len(recommended) / float(n_item)


# --------------------------------------------------------------------------- #
# Efficiency
# --------------------------------------------------------------------------- #
def count_trainable_params(sess=None):
    """Total number of trainable scalar parameters in the current graph.

    Counts the default-graph trainable variables. Call after the model is built
    (a live session is not strictly required, but is accepted for convenience).
    """
    import tensorflow as tf  # lazy import: keep the pure helpers TF-free
    total = 0
    for v in tf.compat.v1.trainable_variables():
        shape = v.shape.as_list()
        if any(s is None for s in shape):
            continue
        n = 1
        for s in shape:
            n *= int(s)
        total += n
    return int(total)


# --------------------------------------------------------------------------- #
# Pareto / hypervolume (for MINIMIZATION problems)
# --------------------------------------------------------------------------- #
def dominates(a, b):
    """True if objective vector ``a`` Pareto-dominates ``b`` (minimization)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.all(a <= b) and np.any(a < b)


def pareto_front_indices(objs):
    """Indices of non-dominated rows in ``objs`` (minimization)."""
    objs = np.asarray(objs, dtype=float)
    n = objs.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        for j in range(n):
            if i != j and dominates(objs[j], objs[i]):
                is_efficient[i] = False
                break
    return np.where(is_efficient)[0]


def hypervolume(objs, ref_point, n_samples=200000, seed=0):
    """Monte-Carlo hypervolume of a non-dominated set (minimization).

    The hypervolume is the volume of the region dominated by the front and
    bounded above by ``ref_point``. A larger value means a better front
    (closer to the ideal point and/or more spread). Monte-Carlo estimation is
    used so the function works for any number of objectives without extra deps.

    Args:
        objs: (m, d) objective vectors to MINIMIZE.
        ref_point: length-d worst-case reference (must be dominated by the front).
        n_samples: number of random points for the estimate.
        seed: RNG seed for reproducibility.
    """
    objs = np.asarray(objs, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    if objs.size == 0:
        return 0.0
    # keep only the Pareto-optimal, in-bounds points
    front = objs[pareto_front_indices(objs)]
    front = front[np.all(front <= ref, axis=1)]
    if front.size == 0:
        return 0.0

    lower = front.min(axis=0)            # ideal corner of the sampling box
    box = np.maximum(ref - lower, 1e-12)
    vol_box = float(np.prod(box))

    rng = np.random.RandomState(seed)
    samples = lower + rng.random((n_samples, front.shape[1])) * box
    # a sample counts if it is dominated by ANY front point (front <= sample)
    dominated = np.zeros(n_samples, dtype=bool)
    for p in front:
        dominated |= np.all(p <= samples, axis=1)
    return vol_box * dominated.mean()


def knee_point_index(objs):
    """Pick a balanced ("knee") solution from a front (minimization).

    Min-max normalize each objective, then choose the point with the smallest
    distance to the ideal (all-zeros) corner. Robust and parameter-free.
    """
    objs = np.asarray(objs, dtype=float)
    idx = pareto_front_indices(objs)
    front = objs[idx]
    lo = front.min(axis=0)
    span = np.maximum(front.max(axis=0) - lo, 1e-12)
    norm = (front - lo) / span
    dist = np.sqrt((norm ** 2).sum(axis=1))
    return int(idx[int(np.argmin(dist))])
