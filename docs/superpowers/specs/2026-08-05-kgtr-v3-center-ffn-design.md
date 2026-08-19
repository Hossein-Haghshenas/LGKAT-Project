# Design: kg_transformer v3 (center query + FFN + temperature)

**Date:** 2026-08-05  
**Branch:** `thesis-transformer`  
**Status:** Approach A approved by AmirAli.  
**Success bar:** mean F1 over 4 datasets ≥ **1.015 ×** current `pool=raw` champion mean (**0.7827 → ≥ 0.7944**).

Baseline champion F1: music 0.7499, book 0.6559, movie 0.9409, restaurant 0.7841.

---

## Problem

Current `--att kg_transformer --kgtr_pool raw` beats paper but user wants a real architectural lift (~+1.5% relative mean F1). Weaknesses of v2 raw:

1. Attention weights = **mean over all query positions** (dilutes center entity signal).
2. No **FFN** — not a full Transformer block after aggregation.
3. No **temperature** — softmax can be too peaky/flat.

## Non-goals

- Do not re-enable failed `kgtr_token product` or `kgtr_pool transformed` as default.
- Do not change paper defaults / seeds / byte-for-byte default path.
- Do not launch training (Hossein only).

## Design (flag-gated, all OFF by default)

| Flag | Default | Effect |
|------|---------|--------|
| `--kgtr_center` | `0` | If `1`: Q from **center `e_i` + user** only; K from neighbor tokens; weights = mean-over-heads of that single query. Else: legacy mean-over-queries (v2). |
| `--kgtr_ffn` | `0` | If `1`: after raw (or transformed) pool, `agg' = LN(agg + W2·ReLU(W1·agg))`. |
| `--kgtr_temp` | `1.0` | Divide attention logits by `temp` before softmax (`temp>0`). |

Keep: `--kgtr_token rel_ej`, `--kgtr_pool raw` for champion smoke.

### Center query (detail)

```
tok_j = W_in(concat(r⊙ei, ej))           # same as v2
Q = W_q(ei) + W_u(u)                        # [N, 1, d] then split heads
K = W_k(tok)                                # [N, T, d]
logits = Q K^T / (√d_h · temp)              # [N, H, 1, T]
w = mean_heads(softmax(logits))             # [N, T]
agg = mean(w · ej)                          # raw pool, Product-compatible scale
```

### FFN (detail)

```
h1 = ReLU(agg W1 + b1)   # W1: [d, d]
agg' = LN(agg + (h1 W2 + b2))
```

Variables created **only if** `kgtr_ffn==1`: `att_ffn_w1/b1`, `att_ffn_w2/b2`, and reuse or add `att_ffn_ln_*` (separate from block LN of transformed pool).

### L2

Include new `att_ffn_w*` in existing `att_w*` L2 loop (name prefix `att_ffn_w` or `att_w_ffn`).

## Files

- `src/aggregators.py` — forward
- `src/model.py` — flags, weights, asserts
- `src/main.py` — CLI + logging list
- `docs/THESIS_KG_TRANSFORMER_ROADMAP.md` — Phase 5
- `docs/BEST_RESULTS_kg_transformer.md` — stub for v3 tags

## Smoke (music first)

`--kgtr_pool raw --kgtr_center 1 --kgtr_ffn 1 --kgtr_temp 1.0`  
`--exp_tag thesis_kgtr_v3_center_ffn_es25`  
(same music stack as champion).

Then book / movie / restaurant with same center+ffn knobs on their proposed stacks.

## Risks

- FFN on restaurant `d=8` may overfit → if F1 drops, ablate `--kgtr_ffn 0` keep center only.
- +1.5% mean F1 is aggressive on movie; report honest mean even if short.
