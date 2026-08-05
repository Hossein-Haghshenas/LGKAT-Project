# Best results — `kg_transformer` thesis track (`thesis-transformer`)

Snapshot: **2026-08-04**. Protocol: `--runs 3`, `--topk_user_num 100`,
`--att kg_transformer` + **`--kgtr_pool raw`** (QKV scores, mix raw `e_j`).

Design / roadmap:
- [THESIS_KG_TRANSFORMER_ROADMAP.md](THESIS_KG_TRANSFORMER_ROADMAP.md)
- [superpowers/specs/2026-07-31-kg-transformer-design.md](superpowers/specs/2026-07-31-kg-transformer-design.md)

Compare to paper Product, [BEST_RESULTS_proposed.md](BEST_RESULTS_proposed.md),
and [BEST_RESULTS_product_mh.md](BEST_RESULTS_product_mh.md).

---

## Cross-dataset scoreboard (Wave 2 complete)

| Dataset | Best tag | F1 kgtr / paper / proposed / product_mh | R@10 kgtr / paper / proposed / product_mh | vs paper | Notes |
|---------|----------|------------------------------------------|---------------------------------------------|----------|-------|
| **music** | **`thesis_kgtr_v2_pool_raw_es25`** | **0.7499** / 0.7418 / 0.7522 / 0.7525 | **0.1580** / 0.1407 / 0.1648 / 0.1564 | **WIN** | ≈ proposed; R@10 ≥ product_mh |
| **book** | **`thesis_kgtr_book_pool_raw_gate_es`** | **0.6559** / 0.6495 / 0.6559 / 0.6549 | **0.0708** / 0.0696 / 0.0635 / 0.0604 | **WIN** | F1 = proposed; **R@10 beats all three** |
| **movie** | **`thesis_kgtr_movie_pool_raw_fixed_bs8k`** | **0.9409** / 0.9375 / 0.9410 / 0.9408 | **0.2032** / 0.1966 / 0.2014 / 0.2081 | **WIN*** | R@2 slight ↓ vs paper; else ↑ |
| **restaurant** | **`thesis_kgtr_restaurant_pool_raw_fixed_bs32k`** | **0.7841** / 0.7816 / 0.7850 / 0.7855 | **0.1473** / 0.1406 / 0.1498 / 0.1233 | **WIN** | Beats paper R@; ≫ product_mh R@10; ≈ proposed |

**Overall thesis read:** With `--kgtr_pool raw`, `kg_transformer` **beats paper Product on F1 and R@10 on all four datasets** (movie caveat: R@2 only). Ranking on restaurant finally recovered vs the failed `product_mh` / early kgtr-v1 path. Vs proposed champion: competitive everywhere; book R@10 is a clear win over proposed.

Locked recipe knobs: `--att kg_transformer --kgtr_token rel_ej --kgtr_pool raw` (+ per-dataset proposed training stack).

Rejected: v1 `transformed` pool, P1 `kgtr_token product` (music only).

---

## Music — `thesis_kgtr_v2_pool_raw_es25`

| Metric | kgtr | Paper | proposed | product_mh | Δ paper |
|--------|------|-------|----------|------------|---------|
| F1 | **0.7499 ± 0.0009** | 0.7418 | 0.7522 | 0.7525 | **+0.0081** |
| R@10 | **0.1580 ± 0.0316** | 0.1407 | 0.1648 | 0.1564 | **+0.0173** |
| R@100 | **0.4240 ± 0.0198** | 0.4009 | 0.4124 | — | **+0.0231** |

Stack: gate + CL + SimGCL + `early_stop_min_epoch 25`.

---

## Book — `thesis_kgtr_book_pool_raw_gate_es`

| Metric | kgtr | Paper | proposed | product_mh | Δ paper |
|--------|------|-------|----------|------------|---------|
| F1 | **0.6559 ± 0.0008** | 0.6495 | 0.6559 | 0.6549 | **+0.0064** |
| R@10 | **0.0708 ± 0.0201** | 0.0696 | 0.0635 | 0.0604 | **+0.0012** |

Fixes proposed’s historical book MIXED (R@10). High R@10 variance across runs.

---

## Movie — `thesis_kgtr_movie_pool_raw_fixed_bs8k`

| Metric | kgtr | Paper | proposed | product_mh | Δ paper |
|--------|------|-------|----------|------------|---------|
| F1 | **0.9409 ± 0.0003** | 0.9375 | 0.9410 | 0.9408 | **+0.0034** |
| R@2 | 0.0703 ± 0.0209 | 0.0758 | 0.0800 | — | −0.0055 |
| R@10 | **0.2032 ± 0.0345** | 0.1966 | 0.2014 | 0.2081 | **+0.0066** |
| R@100 | **0.6049 ± 0.0099** | 0.5874 | 0.6101 | — | **+0.0175** |

---

## Restaurant — `thesis_kgtr_restaurant_pool_raw_fixed_bs32k`

**Result:** `result/KGCN_LightGCN_restaurant_kg_transformer_thesis_kgtr_restaurant_pool_raw_fixed_bs32k.txt`  
**Per-run JSON:** `result/runs_KGCN_LightGCN_restaurant_kg_transformer_thesis_kgtr_restaurant_pool_raw_fixed_bs32k.json`

| Metric | kgtr (mean±std) | Paper | proposed | product_mh | Δ paper |
|--------|-----------------|-------|----------|------------|---------|
| AUC | 0.8730 ± 0.0005 | — | 0.8728 | — | — |
| F1 | **0.7841 ± 0.0014** | 0.7816 | 0.7850 | 0.7855 | **+0.0025** |
| R@2 | **0.0497 ± 0.0315** | 0.0385 | 0.0468 | — | **+0.0112** |
| R@10 | **0.1473 ± 0.0282** | 0.1406 | 0.1498 | 0.1233 | **+0.0067** |
| R@50 | **0.3681 ± 0.0359** | 0.3440 | 0.3999 | — | **+0.0241** |
| R@100 | **0.5358 ± 0.0339** | 0.5211 | 0.5324 | — | **+0.0147** |

Per-run R@10: **0.175 / 0.158 / 0.109** — variance like proposed restaurant; mean still beats paper.
Early-stop ~ep 6–7 (best eval R@10 ep 1/2/1).

**Verdict:** **WIN vs paper** on F1 and all Recall@K. Slightly under proposed on F1/R@10/R@50;
**far above** `product_mh` R@10 (0.123). Strong recovery of the hardest dataset.

**Recipe:**

```powershell
cd D:\projects\current-git-project\LGKAT-Project\src
conda activate lgkat

python main.py --dataset restaurant --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 32768 --l2_weight 1e-7 --lr 0.02 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att kg_transformer --n_heads 2 --kgtr_token rel_ej --kgtr_pool raw --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag thesis_kgtr_restaurant_pool_raw_fixed_bs32k
```

---

## Rejected / weak tags

| Tag | Why |
|-----|-----|
| `thesis_kgtr_v3stack_es25` (v1 transformed) | Music ranking below paper |
| `thesis_kgtr_v2_token_u_es25` (P1) | Music F1 ≤ paper |

---

## Next (optional)

- Paired significance vs proposed where close (`scripts/significance_test.py`).
- Thesis tables: paper vs proposed vs `product_mh` vs `kg_transformer` (`pool=raw`).
- Optional `--topk_user_num 0` confirm for final claims.
- Commit result files + docs with the code that produced them.

---

## Claim caveats

`topk_user_num 100`; movie/restaurant batch ≠ paper Table 1; restaurant R@ std high.
