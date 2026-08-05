# Design: `--att kg_transformer` (MH-B v1)

**Date:** 2026-07-31  
**Branch:** `thesis-transformer`  
**Status:** Approved by AmirAli (approach A).  
**Roadmap:** [THESIS_KG_TRANSFORMER_ROADMAP.md](../../THESIS_KG_TRANSFORMER_ROADMAP.md)

---

## Problem

Proposal title requires Transformer / multi-head KG attention. Phase 3a shipped
`--att product_mh` (multi-head **product scoring** only). Ranking still weak on
book/restaurant vs paper. Need a **literal** Graph Transformer block over fixed KG
neighbors, flag-gated, without breaking default/product reproduction.

---

## Goals / non-goals

**Goals**

- One Transformer-style block: multi-head QKV self-attention + residual + LayerNorm.
- Personalized via **user-conditioned queries** (LGKAT story).
- Reuse `ConcatAggregator` and existing training stack (`--fusion`, `--use_cl`, …).
- OFF unless `--att kg_transformer`; default path byte-for-byte unchanged.
- Export neighbor attention for `--interpret`.

**Non-goals (v1)**

- FFN / full encoder layer (Phase 4 P2).
- Merging `with-multi-head-attention`.
- Unified UI+KG graph, MoE, LLM entity init.
- Changing paper defaults or `*_final` data.

---

## Approaches considered

| ID | Idea | Decision |
|----|------|----------|
| A | Personalized QKV + residual + LN, mean-pool | **Chosen** |
| B | Pure self-attn, no user in Q | Rejected for v1 (weaker personalization) |
| C | A + FFN | Deferred to Phase 4 if v1 weak |

---

## Architecture (v1)

### Placement

Inside `Aggregator._mix_neighbor_vectors` when `weights_att['att'] == 'kg_transformer'`.
Early-return `neighbors_aggregated`; do **not** fall through to
`softmax_scores * neighbor_vectors` (product path). Downstream `ConcatAggregator`
unchanged: `concat(self_vectors, neighbors_agg)`.

### Token (per neighbor j)

**v1 (`--kgtr_token rel_ej`, default):**

```
t_j = W_in · concat(r_j ⊙ e_i, e_j) + b_in     # ℝ^d ; W_in ∈ R^{2d×d}
```

**Phase-4 P1 (`--kgtr_token product`):**

```
t_j = W_in · concat(u, r_j ⊙ e_i, e_j) + b_in   # W_in ∈ R^{3d×d}
```

Same Product-style features as paper attention; user still also enters Q via `W_u`.

Shape after build: `[batch, seq, T, d]` with `T = neighbor_sample_size`.

### Multi-head attention

Flatten to `[N, T, d]` where `N = batch * seq`.

```
Q = X W_q + U W_u      # U = user emb tiled to [N, T, d]
K = X W_k
V = X W_v
```

Split into `H` heads (`d_h = d / H`), scores `Q K^T / √d_h`, softmax over keys,
`O = concat(heads) W_o`.

### Block + pool

```
h = LayerNorm(X + O)           # gamma/beta learnable; eps=1e-6; manual TF1 LN
neighbors_agg = mean(h, axis=T)  # [batch, seq, d]
```

Attention dropout = **0** for first smoke.

### Interpret export

`attention_scores`: mean over heads and over **query** axis of the `T×T` attn matrix
→ `[batch, seq, T]` (importance per neighbor), same Task-6 shape as product.

### Variables (create only if `att == kg_transformer`)

| Name | Shape |
|------|-------|
| `att_w_in`, `att_b_in` | `[2d, d]`, `[d]` |
| `att_w_q`, `att_w_k`, `att_w_v`, `att_w_u`, `att_w_o` | `[d, d]` each |
| `att_ln_gamma`, `att_ln_beta` | `[d]` |

L2: all `att_w*` tensors (same pattern as `product_mh`).

### CLI

- `--att kg_transformer`
- `--n_heads` (existing; assert `dim % n_heads == 0`)
- `--exp_tag thesis_kgtr_*`

Suggested heads: music/movie/book `4`; restaurant `2` (`dim=8`).

---

## Files to touch

| File | Change |
|------|--------|
| `src/aggregators.py` | `kg_transformer` branch + early return |
| `src/model.py` | init weights, include in att list + L2 + dim assert |
| `src/main.py` | help text for `att` / `n_heads` |
| Docs | roadmap, this spec, plan, empty scoreboard |

---

## Risks

- Over-parameterization on book / restaurant (`d=8`).
- Higher compute than product (T×T attention; T≤8 — fine).
- Stale branch lessons: keep flag OFF by default; clean rewrite only.

## Success metrics

See roadmap §1. Compare via `scripts/compare_run_to_paper.py`; close calls →
`scripts/significance_test.py`.
