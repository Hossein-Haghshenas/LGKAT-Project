# data-only HHI diagnostic on kg_final; does not train or write *_final
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
DATASETS = ("music", "book", "movie", "restaurant")
NEIGHBOR_N = {"music": 8, "book": 8, "movie": 4, "restaurant": 4}


def load_array(path_no_ext: str) -> np.ndarray:
    npy = path_no_ext + ".npy"
    txt = path_no_ext + ".txt"
    if os.path.exists(npy):
        return np.load(npy)
    return np.loadtxt(txt, dtype=np.int64)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x, kind="mergesort")).astype(np.float64)
    ry = np.argsort(np.argsort(y, kind="mergesort")).astype(np.float64)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def unique_hhi_clusters(hhi: np.ndarray, decimals: int = 8) -> dict:
    rounded = np.round(hhi, decimals)
    vals, counts = np.unique(rounded, return_counts=True)
    n_tied = int(np.sum(counts[counts > 1]))
    n_cluster = int(np.sum(counts > 1))
    return {
        "n_unique_rounded": int(len(vals)),
        "n_relations_in_ties": n_tied,
        "n_tied_clusters": n_cluster,
        "pct_in_ties": float(100.0 * n_tied / len(hhi)) if len(hhi) else 0.0,
    }


def item_coverage(kg_np: np.ndarray, n_item: int, keep_rels: set, n_neighbor: int) -> dict:
    nbr = defaultdict(set)
    for h, r, t in kg_np:
        r = int(r)
        if r not in keep_rels:
            continue
        h, t = int(h), int(t)
        nbr[h].add(t)
        nbr[t].add(h)
    n_ent = int(max(kg_np[:, 0].max(), kg_np[:, 2].max()) + 1)
    item_deg = np.array([len(nbr[i]) for i in range(min(n_item, n_ent))], dtype=np.int64)
    if item_deg.size == 0:
        return {}
    return {
        "n_item": int(item_deg.size),
        "mean_unique_neighbors": float(item_deg.mean()),
        "median_unique_neighbors": float(np.median(item_deg)),
        "pct_items_deg_ge_N": float(100.0 * np.mean(item_deg >= n_neighbor)),
        "pct_items_deg_0": float(100.0 * np.mean(item_deg == 0)),
        "pct_items_deg_lt_N": float(100.0 * np.mean(item_deg < n_neighbor)),
    }


def analyze(dataset: str) -> dict:
    kg = load_array(os.path.join(DATA, dataset, "kg_final"))
    ratings = load_array(os.path.join(DATA, dataset, "ratings_final"))
    n_item = int(len(set(ratings[:, 1].tolist())))
    n_neighbor = NEIGHBOR_N[dataset]

    # directed triples as stored
    rels = np.unique(kg[:, 1])
    rows = []
    for r in rels:
        mask = kg[:, 1] == r
        sub = kg[mask]
        n_r = int(sub.shape[0])
        H = np.unique(sub[:, 0])
        T = np.unique(sub[:, 2])
        tails, counts = np.unique(sub[:, 2], return_counts=True)
        p = counts.astype(np.float64) / n_r
        hhi = float(np.sum(p * p))
        tph = n_r / float(len(H))
        inv_t = 1.0 / float(len(T))
        rows.append(
            {
                "r": int(r),
                "n_r": n_r,
                "n_H": int(len(H)),
                "n_T": int(len(T)),
                "HHI": hhi,
                "inv_T": inv_t,
                "HHI_minus_inv_T": hhi - inv_t,
                "HHI_over_inv_T": hhi / inv_t,
                "tph": float(tph),
            }
        )
    rows.sort(key=lambda z: (-z["HHI"], z["r"]))
    hhi = np.array([z["HHI"] for z in rows], dtype=np.float64)
    n_r_arr = np.array([z["n_r"] for z in rows], dtype=np.float64)
    inv_t_arr = np.array([z["inv_T"] for z in rows], dtype=np.float64)

    n_rel = len(rows)
    n_hhi1 = int(np.sum(np.abs(hhi - 1.0) < 1e-12))
    n_t1 = int(sum(1 for z in rows if z["n_T"] == 1))
    n_hhi_eq_inv = int(sum(1 for z in rows if abs(z["HHI_minus_inv_T"]) < 1e-12))

    ties = unique_hhi_clusters(hhi)

    total_triples = int(kg.shape[0])
    keep_curves = []
    for p in (1.0, 0.75, 0.5, 0.25):
        k = max(1, int(np.ceil(p * n_rel))) if p < 1.0 else n_rel
        kept_types = [z["r"] for z in rows[:k]]
        keep_set = set(kept_types)
        n_kept_triples = int(sum(z["n_r"] for z in rows if z["r"] in keep_set))
        cov = item_coverage(kg, n_item, keep_set, n_neighbor)
        keep_curves.append(
            {
                "p_types": p,
                "n_types_kept": k,
                "triple_keep_frac": n_kept_triples / total_triples,
                "n_triples_kept": n_kept_triples,
                "coverage": cov,
            }
        )

    # mass-based: add relations in HHI order until >= p of triples
    mass_curves = []
    for p in (0.75, 0.5, 0.25):
        acc = 0
        kept = []
        for z in rows:
            kept.append(z["r"])
            acc += z["n_r"]
            if acc / total_triples >= p:
                break
        keep_set = set(kept)
        cov = item_coverage(kg, n_item, keep_set, n_neighbor)
        mass_curves.append(
            {
                "p_mass_target": p,
                "n_types_kept": len(kept),
                "triple_keep_frac": acc / total_triples,
                "n_triples_kept": acc,
                "coverage": cov,
            }
        )

    full_cov = item_coverage(kg, n_item, set(int(r) for r in rels), n_neighbor)

    return {
        "dataset": dataset,
        "n_triples": total_triples,
        "n_relations": n_rel,
        "n_item": n_item,
        "n_entity": int(len(set(kg[:, 0].tolist()) | set(kg[:, 2].tolist()))),
        "neighbor_sample_size": n_neighbor,
        "hhi_min": float(hhi.min()),
        "hhi_median": float(np.median(hhi)),
        "hhi_max": float(hhi.max()),
        "hhi_std": float(hhi.std(ddof=0)),
        "hhi_mean": float(hhi.mean()),
        "n_HHI_eq_1": n_hhi1,
        "pct_HHI_eq_1": float(100.0 * n_hhi1 / n_rel),
        "n_T_eq_1": n_t1,
        "pct_T_eq_1": float(100.0 * n_t1 / n_rel),
        "n_HHI_eq_1_over_T": n_hhi_eq_inv,
        "pct_HHI_eq_uniform": float(100.0 * n_hhi_eq_inv / n_rel),
        "ties": ties,
        "spearman_hhi_inv_T": spearman(hhi, inv_t_arr),
        "spearman_hhi_n_r": spearman(hhi, n_r_arr),
        "pearson_hhi_inv_T": float(np.corrcoef(hhi, inv_t_arr)[0, 1]) if n_rel > 1 else float("nan"),
        "full_coverage": full_cov,
        "keep_by_type_fraction": keep_curves,
        "keep_by_triple_mass": mass_curves,
        "ranking": rows,
    }


def main():
    out = {ds: analyze(ds) for ds in DATASETS}
    dest = os.path.join(ROOT, "scripts", "_hhi_kg_diagnostic.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", dest)


if __name__ == "__main__":
    main()
