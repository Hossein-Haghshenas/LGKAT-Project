"""
moo_core.py
===========
A small, self-contained NSGA-II for MIXED-variable, MULTI-objective problems.

Why hand-rolled instead of pymoo? The optimizer has to drive a TensorFlow-1.x
training loop and mix categorical / integer / log-uniform / binary-vector genes.
A dependency-free implementation keeps the experiment reproducible and avoids
pymoo API drift across versions. The algorithm itself is the standard NSGA-II
(Deb et al., 2002): fast non-dominated sorting + crowding-distance selection,
binary tournament, SBX crossover and polynomial mutation for real genes, and
uniform / bit-flip operators for discrete genes. The optimizer MINIMIZES every
objective.

Search-space spec: a list of gene dicts, e.g.
    {'name': 'dim',  'type': 'cat',   'choices': [8, 16, 32, 64]}
    {'name': 'lr',   'type': 'float', 'low': 1e-4, 'high': 5e-2, 'log': True}
    {'name': 'nlay', 'type': 'int',   'low': 1, 'high': 4}
    {'name': 'rmask','type': 'binary','length': 60, 'min_active': 1}

A genome is a dict {gene_name: value}; binary genes hold a numpy int array.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Per-gene operators
# --------------------------------------------------------------------------- #
def _sample_gene(g, rng):
    t = g['type']
    if t == 'cat':
        return g['choices'][rng.randint(len(g['choices']))]
    if t == 'int':
        return int(rng.randint(g['low'], g['high'] + 1))
    if t == 'float':
        if g.get('log'):
            lo, hi = np.log10(g['low']), np.log10(g['high'])
            return float(10 ** (lo + rng.random() * (hi - lo)))
        return float(g['low'] + rng.random() * (g['high'] - g['low']))
    if t == 'binary':
        mask = (rng.random(g['length']) < g.get('p_keep', 0.6)).astype(np.int64)
        return _repair_binary(mask, g.get('min_active', 1), rng)
    raise ValueError('unknown gene type: %s' % t)


def _repair_binary(mask, min_active, rng):
    mask = (np.asarray(mask) > 0).astype(np.int64)
    if mask.sum() < min_active:
        idx = rng.choice(len(mask), size=min_active, replace=False)
        mask[idx] = 1
    return mask


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _sbx(x1, x2, lo, hi, eta, rng):
    """Simulated binary crossover for one real variable."""
    if abs(x1 - x2) < 1e-12:
        return x1, x2
    u = rng.random()
    if u <= 0.5:
        beta = (2 * u) ** (1.0 / (eta + 1))
    else:
        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
    c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
    c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
    return _clip(c1, lo, hi), _clip(c2, lo, hi)


def _poly_mut(x, lo, hi, eta, rng):
    """Polynomial mutation for one real variable."""
    if hi <= lo:
        return x
    delta1 = (x - lo) / (hi - lo)
    delta2 = (hi - x) / (hi - lo)
    u = rng.random()
    mut_pow = 1.0 / (eta + 1)
    if u < 0.5:
        xy = 1.0 - delta1
        val = 2 * u + (1 - 2 * u) * (xy ** (eta + 1))
        deltaq = val ** mut_pow - 1.0
    else:
        xy = 1.0 - delta2
        val = 2 * (1 - u) + 2 * (u - 0.5) * (xy ** (eta + 1))
        deltaq = 1.0 - val ** mut_pow
    return _clip(x + deltaq * (hi - lo), lo, hi)


def _crossover_gene(g, v1, v2, eta_c, rng):
    t = g['type']
    if t == 'cat' or t == 'int':
        # uniform: each child inherits one parent's value
        if rng.random() < 0.5:
            return v1, v2
        return v2, v1
    if t == 'float':
        if g.get('log'):
            lo, hi = np.log10(g['low']), np.log10(g['high'])
            c1, c2 = _sbx(np.log10(v1), np.log10(v2), lo, hi, eta_c, rng)
            return float(10 ** c1), float(10 ** c2)
        c1, c2 = _sbx(v1, v2, g['low'], g['high'], eta_c, rng)
        return c1, c2
    if t == 'binary':
        cross = rng.random(g['length']) < 0.5
        ch1 = np.where(cross, v1, v2).astype(np.int64)
        ch2 = np.where(cross, v2, v1).astype(np.int64)
        ma = g.get('min_active', 1)
        return _repair_binary(ch1, ma, rng), _repair_binary(ch2, ma, rng)
    raise ValueError(t)


def _mutate_gene(g, v, eta_m, rng):
    t = g['type']
    if t == 'cat':
        return g['choices'][rng.randint(len(g['choices']))]
    if t == 'int':
        return int(rng.randint(g['low'], g['high'] + 1))
    if t == 'float':
        if g.get('log'):
            lo, hi = np.log10(g['low']), np.log10(g['high'])
            return float(10 ** _poly_mut(np.log10(v), lo, hi, eta_m, rng))
        return _poly_mut(v, g['low'], g['high'], eta_m, rng)
    if t == 'binary':
        flip = rng.random(g['length']) < (1.0 / g['length'])
        nv = np.where(flip, 1 - v, v).astype(np.int64)
        return _repair_binary(nv, g.get('min_active', 1), rng)
    raise ValueError(t)


# --------------------------------------------------------------------------- #
# NSGA-II machinery
# --------------------------------------------------------------------------- #
def _dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(objs):
    """Return a list of fronts; each front is a list of indices into ``objs``."""
    n = len(objs)
    S = [[] for _ in range(n)]
    nd = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                S[p].append(q)
            elif _dominates(objs[q], objs[p]):
                nd[p] += 1
        if nd[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(objs, front):
    """Crowding distance for the indices in ``front``."""
    l = len(front)
    dist = {i: 0.0 for i in front}
    if l == 0:
        return dist
    objs = np.asarray(objs, dtype=float)
    d = objs.shape[1]
    for m in range(d):
        order = sorted(front, key=lambda i: objs[i, m])
        dist[order[0]] = float('inf')
        dist[order[-1]] = float('inf')
        fmin, fmax = objs[order[0], m], objs[order[-1], m]
        span = fmax - fmin
        if span <= 1e-12:
            continue
        for k in range(1, l - 1):
            dist[order[k]] += (objs[order[k + 1], m] - objs[order[k - 1], m]) / span
    return dist


def _assign_rank_crowd(pop):
    """Annotate each individual with 'rank' and 'crowd' in-place."""
    objs = [ind['obj'] for ind in pop]
    fronts = fast_non_dominated_sort(objs)
    for r, front in enumerate(fronts):
        cd = crowding_distance(objs, front)
        for i in front:
            pop[i]['rank'] = r
            pop[i]['crowd'] = cd[i]
    return fronts


def _tournament(pop, rng):
    a, b = pop[rng.randint(len(pop))], pop[rng.randint(len(pop))]
    if a['rank'] != b['rank']:
        return a if a['rank'] < b['rank'] else b
    return a if a['crowd'] >= b['crowd'] else b


def _make_offspring(pop, space, pop_size, cx_prob, eta_c, eta_m, rng):
    offspring = []
    while len(offspring) < pop_size:
        p1 = _tournament(pop, rng)['genome']
        p2 = _tournament(pop, rng)['genome']
        c1, c2 = {}, {}
        for g in space:
            name = g['name']
            if rng.random() < cx_prob:
                v1, v2 = _crossover_gene(g, p1[name], p2[name], eta_c, rng)
            else:
                v1, v2 = p1[name], p2[name]
            c1[name], c2[name] = v1, v2
        # mutation (gene-level probability = 1/num_genes)
        m_prob = 1.0 / len(space)
        for child in (c1, c2):
            for g in space:
                if rng.random() < m_prob:
                    child[g['name']] = _mutate_gene(g, child[g['name']], eta_m, rng)
        offspring.append({'genome': c1, 'obj': None})
        if len(offspring) < pop_size:
            offspring.append({'genome': c2, 'obj': None})
    return offspring


def run_nsga2(evaluate_fn, space, pop_size=20, n_gen=15, seed=0,
              ref_point=None, cx_prob=0.9, eta_c=15, eta_m=20,
              verbose=True, on_generation=None):
    """Run NSGA-II.

    Args:
        evaluate_fn: genome(dict) -> (obj_vector(list/array to MINIMIZE), info(dict)).
        space: list of gene specs (see module docstring).
        pop_size, n_gen, seed: GA budget / reproducibility.
        ref_point: optional reference point for the hypervolume history.
        on_generation: optional callback(gen, pop, archive) for logging.

    Returns:
        dict with 'archive' (all evaluated individuals), 'final_pop',
        'history' (list of per-generation stats).
    """
    from objectives import hypervolume, pareto_front_indices

    rng = np.random.RandomState(seed)
    history = []
    archive = []

    def _evaluate(ind):
        obj, info = evaluate_fn(ind['genome'])
        ind['obj'] = np.asarray(obj, dtype=float)
        ind['info'] = info
        archive.append(ind)

    # initial population
    pop = [{'genome': {g['name']: _sample_gene(g, rng) for g in space}, 'obj': None}
           for _ in range(pop_size)]
    for ind in pop:
        _evaluate(ind)
    _assign_rank_crowd(pop)

    for gen in range(n_gen):
        offspring = _make_offspring(pop, space, pop_size, cx_prob, eta_c, eta_m, rng)
        for ind in offspring:
            _evaluate(ind)
        combined = pop + offspring
        fronts = _assign_rank_crowd(combined)

        # environmental selection: fill next pop front by front
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(combined[i] for i in front)
            else:
                cd = crowding_distance([c['obj'] for c in combined], front)
                ordered = sorted(front, key=lambda i: cd[i], reverse=True)
                need = pop_size - len(new_pop)
                new_pop.extend(combined[i] for i in ordered[:need])
                break
        pop = new_pop
        _assign_rank_crowd(pop)

        # stats
        objs = np.array([ind['obj'] for ind in pop])
        hv = None
        if ref_point is not None:
            hv = hypervolume(objs, ref_point)
        n_front0 = int(np.sum([ind['rank'] == 0 for ind in pop]))
        stat = {'gen': gen, 'hypervolume': hv, 'n_front0': n_front0,
                'best_per_obj': objs.min(axis=0).tolist()}
        history.append(stat)
        if verbose:
            hv_str = ('%.5f' % hv) if hv is not None else 'NA'
            print('[NSGA-II] gen %2d/%d  HV=%s  front0=%d  best_obj=%s'
                  % (gen + 1, n_gen, hv_str, n_front0,
                     np.round(objs.min(axis=0), 4).tolist()))
        if on_generation is not None:
            on_generation(gen, pop, archive)

    return {'archive': archive, 'final_pop': pop, 'history': history}
