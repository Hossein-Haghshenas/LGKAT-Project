# Best results — `product_mh` thesis track (`thesis-transformer`)

Snapshot: **2026-07-30**. Protocol: `--runs 3`, `--topk_user_num 100`,
`--att product_mh` (channel-split, scaled mean logits).

Compare to paper Product and [BEST_RESULTS_proposed.md](BEST_RESULTS_proposed.md).

---

## Cross-dataset scoreboard (complete)

| Dataset | Best MH tag | F1 MH / paper / proposed | R@10 MH / paper / proposed | vs paper | vs proposed |
|---------|-------------|--------------------------|----------------------------|----------|-------------|
| **music** | `thesis_pmh_v3stack_es25` | **0.7525** / 0.7418 / 0.7522 | **0.1564** / 0.1407 / 0.1648 | **WIN** | ≈ F1; R@10 slight ↓ |
| **book** | `thesis_pmh_book_gate_es` | **0.6549** / 0.6495 / 0.6559 | 0.0604 / 0.0696 / 0.0635 | **MIXED** | slight ↓ |
| **movie** | `thesis_pmh_movie_fixed_bs8k` | **0.9408** / 0.9375 / 0.9410 | **0.2081** / 0.1966 / 0.2014 | **WIN*** | ≈ F1; R@10 ↑ |
| **restaurant** | `thesis_pmh_restaurant_fixed_bs32k` | **0.7855** / 0.7816 / 0.7850 | 0.1233 / 0.1406 / 0.1498 | **MIXED** | ≈ F1; **R@10 ↓** |
| restaurant (retry) | `…_fixed_bs32k_es8` | 0.7838 / 0.7816 / 0.7850 | 0.1231 / 0.1406 / 0.1498 | MIXED, worse F1 | early-stop chase **failed** — do not prefer |

\*Movie: F1/R@10/R@100 ↑ vs paper; R@2/R@50 slight ↓.

**Overall thesis read:** `product_mh` **beats or matches paper Product on F1 on all four datasets**. Ranking vs paper is **WIN on music/movie**, **MIXED on book/restaurant** (R@10 below paper). Vs proposed champion: near-tie on F1 everywhere; ranking competitive on movie, slightly behind elsewhere — restaurant ranking is the weakest transfer.

---

## Restaurant — `thesis_pmh_restaurant_fixed_bs32k`

Same as `proposed_restaurant_fixed_bs32k` but `--att product_mh --n_heads 2` (`dim=8`).

**Result:** `result/KGCN_LightGCN_restaurant_product_mh_thesis_pmh_restaurant_fixed_bs32k.txt`  
**Per-run JSON:** `result/runs_KGCN_LightGCN_restaurant_product_mh_thesis_pmh_restaurant_fixed_bs32k.json`

| Metric | MH (mean±std) | Paper | proposed | Δ paper | Δ proposed |
|--------|---------------|-------|----------|---------|------------|
| AUC | 0.8728 ± 0.0005 | — | 0.8728 | — | 0.0000 |
| F1 | **0.7855 ± 0.0002** | 0.7816 | 0.7850 | **+0.0039** | **+0.0005** |
| R@2 | 0.0351 ± 0.0229 | 0.0385 | 0.0468 | −0.0034 | −0.0117 |
| R@10 | 0.1233 ± 0.0221 | 0.1406 | 0.1498 | −0.0173 | −0.0265 |
| R@50 | 0.3769 ± 0.0355 | 0.3440 | 0.3999 | **+0.0329** | −0.0230 |
| R@100 | 0.5140 ± 0.0419 | 0.5211 | 0.5324 | −0.0071 | −0.0184 |

Per-run R@10: 0.151 / 0.122 / 0.097 — high variance (same class of noise as proposed restaurant). Run 2 early-stopped with best eval epoch 0.

**Verdict:** CTR fine (F1 ≥ paper and ≈ proposed). Ranking **weaker than paper R@10 and proposed** — honest MIXED / under-champion on Recall. Not a training collapse.

**Recipe:**

```powershell
cd D:\projects\current-git-project\LGKAT-Project\src
conda activate lgkat

python main.py --dataset restaurant --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 32768 --l2_weight 1e-7 --lr 0.02 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att product_mh --n_heads 2 --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag thesis_pmh_restaurant_fixed_bs32k
```

---

## Music / book / movie (short)

| Dataset | Tag | F1 | R@10 | Note |
|---------|-----|-----|------|------|
| music | `thesis_pmh_v3stack_es25` | 0.7525 | 0.1564 | Best MH stack; need `early_stop_min_epoch 25` |
| book | `thesis_pmh_book_gate_es` | 0.6549 | 0.0604 | MIXED like proposed |
| movie | `thesis_pmh_movie_fixed_bs8k` | 0.9408 | 0.2081 | R@10 beats proposed |

---

## Optional follow-ups (restaurant ranking chase)

### Retry `thesis_pmh_restaurant_fixed_bs32k_es8` — **rejected**

Same MH (`n_heads 2`) + `--early_stop 8 --early_stop_min_epoch 5`.

| Metric | es8 | prior MH | Paper | proposed |
|--------|-----|----------|-------|----------|
| F1 | 0.7838 ± 0.0024 | **0.7855** | 0.7816 | 0.7850 |
| R@10 | 0.1231 ± 0.0125 | 0.1233 | 0.1406 | 0.1498 |
| R@100 | 0.5335 ± 0.0352 | 0.5140 | 0.5211 | 0.5324 |

R@10 mean unchanged; F1 slightly worse. Early-stop was **not** the bottleneck.
**Keep** `thesis_pmh_restaurant_fixed_bs32k` as restaurant MH best. **Stop** further restaurant hparam chasing unless a new architectural idea (e.g. `n_heads 4`) is explicitly prioritized — default path = freeze table and move on.

1. ~~Re-run with early_stop_min_epoch~~ done, no gain.  
2. Optional only if needed later: `--n_heads 4` ablation.  
3. Restaurant ranking under `topk_user_num 100` stays high-variance (proposed also uneven).

Default recommendation: **freeze the four-dataset table** (`restaurant` best = first MH tag, not es8).
Next step options: (1) git commit of `product_mh` code + `result/*thesis_pmh*` + docs,
(2) thesis ablation table (MH alone vs stack on music), (3) optional `n_heads 4` on restaurant only if you insist on one last chase.

---

## Caveats

- Means over 3 runs; `topk_user_num 100` ≠ paper all-users.  
- Movie/restaurant batch≠paper Table 1.  
- Never overwrite `result/*proposed*`.
