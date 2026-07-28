# Best results — proposed model (`major-version-1`)

Scoreboard for the **proposed** LGKAT variant on this branch: paper Product baseline
plus flag-gated upgrades (fusion gate / fixed, cross-view CL, SimGCL, LR decay, early
stop, `--fast_eval`). Sibling docs for later models: `docs/BEST_RESULTS_<model>.md`.

Snapshot date: **2026-07-28**.  
Paper baselines = Product LGKAT row in Tables 2–3 (`scripts/paper_targets_product.py`).  
Our numbers = mean over `--runs 3`, test set, `--att product`.  
Protocol note: all proposed runs below use `--topk_user_num 100` (fast, higher variance). Paper tables use all eligible users (`0`).

Run from `src/` after `conda activate lgkat`.

---

## Summary scoreboard

| Dataset | Best `exp_tag` | F1 (ours / paper) | R@10 (ours / paper) | Verdict vs paper |
|---------|----------------|-------------------|---------------------|------------------|
| **music** (Last.fm) | `proposed_v3` | **0.7522** / 0.7418 | **0.1648** / 0.1407 | **WIN** (all listed metrics ↑) |
| **book** (Book-Crossing) | `proposed_book_gate_es` | **0.6559** / 0.6495 | 0.0635 / 0.0696 | **MIXED** (F1 / R@2 / R@100 ↑; R@10 slightly ↓) |
| **movie** (MovieLens) | `proposed_movie_fixed_bs8k` | **0.9410** / 0.9375 | **0.2014** / 0.1966 | **WIN** (all ↑) |
| **restaurant** (Dianping) | `proposed_restaurant_fixed_bs32k` | **0.7850** / 0.7816 | **0.1498** / 0.1406 | **WIN** on mean (ranking std high) |

Δ = ours − paper (positive = better for these metrics).

---

## 1. Music — `proposed_v3`

**Result file:** `result/KGCN_LightGCN_music_product_proposed_v3.txt`  
**Per-run JSON:** `result/runs_KGCN_LightGCN_music_product_proposed_v3.json`

| Metric | Ours (mean±std) | Paper | Δ |
|--------|-----------------|-------|---|
| F1 | 0.7522 ± 0.0018 | 0.7418 | **+0.0104** |
| R@2 | 0.0786 ± 0.0170 | 0.0355 | **+0.0431** |
| R@10 | 0.1648 ± 0.0357 | 0.1407 | **+0.0241** |
| R@50 | 0.3415 ± 0.0564 | 0.2936 | **+0.0479** |
| R@100 | 0.4124 ± 0.0318 | 0.4009 | **+0.0115** |
| AUC | 0.8304 ± 0.0002 | — | — |

**Recipe:** paper music flags + fusion gate + light CL/SimGCL + LR decay + early stop.

```powershell
cd D:\projects\current-git-project\LGKAT-Project\src
conda activate lgkat

python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion gate --use_cl 1 --cl_lambda 0.02 --cl_tau 0.2 --use_simgcl 1 --simgcl_lambda 0.02 --simgcl_eps 0.05 --lr_decay 0.98 --early_stop 12 --fast_eval 1 --exp_tag proposed_v3
```

---

## 2. Book — `proposed_book_gate_es`

**Result file:** `result/KGCN_LightGCN_book_product_proposed_book_gate_es.txt`  
**Per-run JSON:** `result/runs_KGCN_LightGCN_book_product_proposed_book_gate_es.json`

| Metric | Ours (mean±std) | Paper | Δ |
|--------|-----------------|-------|---|
| F1 | 0.6559 ± 0.0005 | 0.6495 | **+0.0064** |
| R@2 | 0.0452 ± 0.0080 | 0.0385 | **+0.0067** |
| R@10 | 0.0635 ± 0.0135 | 0.0696 | −0.0061 |
| R@50 | 0.1082 ± 0.0100 | 0.1081 | **+0.0001** |
| R@100 | 0.1481 ± 0.0028 | 0.1380 | **+0.0101** |
| AUC | 0.7143 ± 0.0001 | — | — |

**Recipe:** paper book dims/S + **lr 2e-4** (4× paper 5e-5) + fusion gate + LR decay + early stop; **no** CL/SimGCL.

```powershell
cd D:\projects\current-git-project\LGKAT-Project\src
conda activate lgkat

python main.py --dataset book --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 0.0002 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion gate --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag proposed_book_gate_es
```

---

## 3. Movie — `proposed_movie_fixed_bs8k`

**Result file:** `result/KGCN_LightGCN_movie_product_proposed_movie_fixed_bs8k.txt`  
**Per-run JSON:** `result/runs_KGCN_LightGCN_movie_product_proposed_movie_fixed_bs8k.json`

| Metric | Ours (mean±std) | Paper | Δ |
|--------|-----------------|-------|---|
| F1 | 0.9410 ± 0.0005 | 0.9375 | **+0.0035** |
| R@2 | 0.0800 ± 0.0182 | 0.0758 | **+0.0042** |
| R@10 | 0.2014 ± 0.0283 | 0.1966 | **+0.0048** |
| R@50 | 0.4641 ± 0.0117 | 0.4528 | **+0.0113** |
| R@100 | 0.6101 ± 0.0107 | 0.5874 | **+0.0227** |
| AUC | 0.9806 ± 0.0000 | — | — |

**Recipe:** paper movie flags but **batch 8192** (paper 65536 OOMs on 16GB) + **lr 0.0025** (linear scale from 0.02) + fixed fusion + LR decay + early stop. Gate + lr 0.02 @8192 underperformed — do not use.

```powershell
cd D:\projects\current-git-project\LGKAT-Project\src
conda activate lgkat

python main.py --dataset movie --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 8192 --l2_weight 5e-6 --lr 0.0025 --layer_size [32] --adj_type plain --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 1 --pretrain 0 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 8 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag proposed_movie_fixed_bs8k
```

---

## 4. Restaurant — `proposed_restaurant_fixed_bs32k`

**Result file:** `result/KGCN_LightGCN_restaurant_product_proposed_restaurant_fixed_bs32k.txt`  
**Per-run JSON:** `result/runs_KGCN_LightGCN_restaurant_product_proposed_restaurant_fixed_bs32k.json`

| Metric | Ours (mean±std) | Paper | Δ |
|--------|-----------------|-------|---|
| F1 | 0.7850 ± 0.0012 | 0.7816 | **+0.0034** |
| R@2 | 0.0468 ± 0.0258 | 0.0385 | **+0.0083** |
| R@10 | 0.1498 ± 0.0330 | 0.1406 | **+0.0092** |
| R@50 | 0.3999 ± 0.0510 | 0.3440 | **+0.0559** |
| R@100 | 0.5324 ± 0.0503 | 0.5211 | **+0.0113** |
| AUC | 0.8728 ± 0.0010 | — | — |

**Recipe:** paper restaurant dims/α/S + **batch 32768** + **lr 0.02** (paper 65536 OOMs; movie’s 8192+0.0025 overfits hard here) + fixed fusion + LR decay + early stop. UI adj is sparse-built and cached under `data/restaurant/s_*.npz` — free ~4GB+ RAM on first build.

Per-run R@10 was uneven (~0.176 / 0.170 / 0.103); treat ranking wins as provisional until a repeat or `--topk_user_num 0`.

```powershell
cd D:\projects\current-git-project\LGKAT-Project\src
conda activate lgkat

python main.py --dataset restaurant --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 32768 --l2_weight 1e-7 --lr 0.02 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag proposed_restaurant_fixed_bs32k
```

---

## Paper reproduction commands (baseline, no proposed flags)

Canonical paper-style flags live in `src/cmd.sh`. Defaults in `main.py` are **not** paper settings — always pass the full flag set.

| Dataset | Paper batch / lr / α / S (from `cmd.sh`) |
|---------|------------------------------------------|
| movie | 65536 / 0.02 / α=0 / S=1 |
| book | 256 / 5e-5 / α=0 / S=3 |
| music | 128 / 0.005 / α=0.5 / S=8 |
| restaurant | 65536 / 0.02 / α=0.5 / S=1 |

Compare any aggregate log:

```powershell
cd D:\projects\current-git-project\LGKAT-Project
python scripts/compare_run_to_paper.py result/KGCN_LightGCN_<dataset>_product_<exp_tag>.txt <dataset>
```

---

## Rejected / weaker variants (do not treat as best)

| Dataset | `exp_tag` | Why discarded |
|---------|-----------|---------------|
| music | `proposed_v2` | Beats some Recall but F1 slightly under paper; v3 strictly better |
| book | `proposed_v3` / `proposed_v3b` | Collapsed (AUC ~0.52, Recall@K = 0) |
| movie | `proposed_movie_gate_es_bs8k` | Under paper on F1 and all Recall@K |
| restaurant | `proposed_restaurant_fixed_bs8k` | Severe CTR overfit (eval AUC drop after epoch 1) |

---

## Caveats when claiming improvements

1. Means are over **3 runs**; a single run is not evidence (`scripts/significance_test.py` for close calls).
2. `--topk_user_num 100` ≠ paper all-users protocol.
3. Hardware forced batch/lr changes on movie (8192) and restaurant (32768); values are not identical to Table 1 training setup.
4. Never regenerate `data/*/ratings_final.*` / `kg_final.*` — breaks comparability with every logged result.
