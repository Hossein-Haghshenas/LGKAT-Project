"""
relation_select.py
==================
Utilities for the MOKAR "KG relation selection" gene.

A relation mask is a 0/1 numpy array of length ``n_relation`` (the FULL relation
count of a dataset). ``mask[r] == 1`` means relation id ``r`` is kept when the
knowledge graph adjacency is built (see ``data_loader.construct_kg``).

Selecting a subset of relations lets the optimizer prune noisy / uninformative
relations. This directly targets the limitation acknowledged by the LGKAT base
paper ("performance is highly dependent on the quality and completeness of the
knowledge graph"). Removing relations also shrinks the effective KG and can
reduce neighbour-aggregation cost, which feeds the efficiency objective.
"""

import numpy as np


def n_active(mask):
    """Number of selected relations."""
    return int(np.sum(np.asarray(mask) > 0))


def mask_to_relations(mask):
    """Return the set of selected relation ids."""
    mask = np.asarray(mask)
    return set(int(r) for r in np.where(mask > 0)[0])


def repair_mask(mask, min_active=1, rng=None):
    """Guarantee at least ``min_active`` relations are kept.

    An all-zero mask would leave the KG empty (every entity self-loops), which
    is degenerate. We never silently train on an empty KG; instead we activate
    a few random relations. Returns a new int array (does not mutate input).
    """
    rng = rng or np.random
    mask = (np.asarray(mask) > 0).astype(np.int64)
    if mask.sum() < min_active:
        idx = rng.choice(len(mask), size=min_active, replace=False)
        mask[idx] = 1
    return mask


def random_mask(n_relation, rng=None, p_keep=0.6, min_active=1):
    """Sample a random relation mask, keeping each relation with prob ``p_keep``."""
    rng = rng or np.random
    mask = (rng.random(n_relation) < p_keep).astype(np.int64)
    return repair_mask(mask, min_active=min_active, rng=rng)


def full_mask(n_relation):
    """Mask that keeps every relation (== original LGKAT behaviour)."""
    return np.ones(n_relation, dtype=np.int64)
