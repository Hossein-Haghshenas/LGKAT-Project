# Design: kg_transformer v4 (relation bias on logits)

**Date:** 2026-08-19  
**Branch:** `thesis-transformer`  
**Status:** Approach A approved by AmirAli.  
**Success bar (criterion C):** mean F1 ≈ **0.787–0.789** (+0.6–0.8% vs 0.7827) **and** F1 ≥ current champion on all four datasets.

Champions: music v3 0.7621 (keep), book 0.6559, movie 0.9409, restaurant 0.7841.

---

## Problem

Phase-5 center+FFN lifted **music only**. Extra params after pool hurt book/restaurant. Need a scoring-only upgrade that stays on `pool=raw`.

## Non-goals

- Do not change mix (`pool=raw` locked).
- Do not re-enable `kgtr_token product`, `kgtr_pool transformed`, or stack v3 FFN on book/movie/restaurant.
- Do not change paper defaults / seeds / byte-for-byte default path.
- Do not launch training.

## Design (flag-gated, OFF by default)

| Flag | Default | Effect |
|------|---------|--------|
| `--kgtr_rel_bias` | `0` | If `1`: `logits_j += w_rel · r_j` before softmax. `r_j` = relation embedding (not `r⊙e_i`). `w_rel` shared across heads, **zero-init**. |

```
logits = (Q K^T) / (√d_h · temp) + (w_rel · r)
w = softmax(logits) over neighbors
agg = mean(w · e_j)                 # raw pool
```

Bias broadcasts over heads and query positions (key axis only).

Variable created **only if** `kgtr_rel_bias==1`: `att_w_rel_bias` `[dim]`, zeros. L2 via `att_w*` prefix.

## Smoke

Book first (v3 failure mode), champion stack + `--kgtr_rel_bias 1 --kgtr_center 0 --kgtr_ffn 0 --kgtr_pool raw`.  
`--exp_tag thesis_kgtr_v4_rel_bias_book`

Pass: F1 ≥ 0.6559. Then movie / restaurant / music (music: add bias on v3 or v2; do not regress 0.7621).

## Fallback (Approach B — entity dot)

If `--kgtr_rel_bias` does not lift book: `--kgtr_dot 1` adds `(e_i · e_j) / √d` to logits (zero extra params; distinct from `--kgtr_center`). Do not stack with FFN. Smoke: `--exp_tag thesis_kgtr_v4_dot_book`.

