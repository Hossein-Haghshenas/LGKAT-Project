# Thesis track — KG Transformer roadmap (`thesis-transformer`)

Living map for the **Transformer / multi-head KG attention** thesis gap.
Last updated: **2026-07-31**. Branch: **`thesis-transformer`** (from `major-version-1`).
Evidence champion stays frozen on `major-version-1` — see [BEST_RESULTS_proposed.md](BEST_RESULTS_proposed.md).

Operational rules: [CLAUDE.md](../CLAUDE.md) / [AGENTS.md](../AGENTS.md).
Parent path: [ROADMAP.md](ROADMAP.md). Engineering depth: [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md).

---

## 1. Goal & success bar

**Goal:** Ship a flag-gated **Graph Transformer–style** module over fixed KG neighborhoods
so the proposal title (“Transformer networks”) is honest, and push F1 / Recall@K above
**paper Product LGKAT** (preferably near or above the proposed champion).

| Bar | Criteria |
|-----|----------|
| Minimum (thesis-defensible) | Mean±std over `--runs 3`; beat **paper** on F1; improve **R@10** vs paper on book+restaurant if possible |
| Stretch | Match/beat [BEST_RESULTS_proposed.md](BEST_RESULTS_proposed.md) on music F1+R@10 |
| Acceptable negative | Beats paper on F1, ranking MIXED, but architecture fills the title gap — still publishable with honest tables |

**Protocol (default for scoreboards):** `--runs 3`, `--topk_user_num 100`. Final claims may
re-run with `0` (all users). Never regenerate `*_final` data. Never overwrite proposed
result files. New runs → new `--exp_tag`. Agents never launch training — copy-paste only.

---

## 2. Where we are (timeline)

| Phase | Status | Deliverable |
|-------|--------|-------------|
| Phase 0 — Freeze evidence | **Done** | `major-version-1` + proposed scoreboard |
| Phase 1 — Branch audit | **Done** | Salvage MH idea; reject merge of stale `with-multi-head-attention` |
| Phase 2 — NotebookLM design | **Done** | [research_notes_notebooklm_transformer.md](research_notes_notebooklm_transformer.md) |
| Phase 3a — MH-A `product_mh` | **Done** | Multi-head **product scoring**; scoreboard [BEST_RESULTS_product_mh.md](BEST_RESULTS_product_mh.md) |
| Phase 3b — MH-B `kg_transformer` | **Wave 2 DONE — WIN vs paper ×4** | Champion = `pool=raw` |
| Phase 4 — Improve if weak | **P2 succeeded** (P1 rejected) | Optional: significance / all-users |
| Phase 5 — v3 lift (+1.5% rel mean F1) | **Closed — miss** | Only music wins; mean ≈0.7858 vs target 0.7944 |
| Phase 5 — Thesis write-up | **Later** | Tables, paired tests, interpret cases |

### Phase 3a result (short)

`product_mh` beats paper on **F1 on all four datasets**. Ranking: WIN music/movie;
**MIXED** book/restaurant (R@10 below paper). Near proposed on F1; restaurant R@ weakest.
This is **not** a full Transformer — only Transformer-*style* multi-head scoring.

---

## 3. Locked design — `--att kg_transformer` (v1)

Approved **2026-07-31** (approach A). Full spec:
[superpowers/specs/2026-07-31-kg-transformer-design.md](superpowers/specs/2026-07-31-kg-transformer-design.md).
Implementation plan:
[superpowers/plans/2026-07-31-kg-transformer.md](superpowers/plans/2026-07-31-kg-transformer.md).

**One-liner:** relation-aware neighbor tokens → multi-head QKV self-attention with
**user-conditioned Q** → residual + LayerNorm → mean-pool → existing `ConcatAggregator`.

| Knob | v1 choice |
|------|-----------|
| CLI | `--att kg_transformer` (OFF unless selected); reuse `--n_heads` |
| Default path | Unchanged (`product` / defaults byte-for-byte) |
| Token | `W_in(concat(r⊙e_i, e_j))` |
| Q | `W_q(t) + W_u(u)` |
| K, V | from tokens |
| Block | `LN(t + MHA(t))`; attn dropout **0** for first smoke |
| Pool | mean over T neighbors |
| Heads | `dim % n_heads == 0`; music/movie `H=4`; restaurant `H=2`; book try `4` |
| Scoreboard | [BEST_RESULTS_kg_transformer.md](BEST_RESULTS_kg_transformer.md) |
| Tags | `thesis_kgtr_*` |

**Do not** merge code from `with-multi-head-attention` (ON-by-default, stale).

---

## 4. Experiment plan (Hossein GPU)

Always `cd src` + `conda activate lgkat`. Full flag sets from paper/`cmd_proposed` —
do not rely on argparse defaults.

### Wave 1 — music smoke (order matters)

1. **`thesis_kgtr_mh_only`** — optional isolate (skipped).
2. **`thesis_kgtr_v3stack_es25`** (v1 transformed) — MIXED / weak ranking.
3. P1 `token=product` — **rejected**.
4. **`thesis_kgtr_v2_pool_raw_es25`** — **DONE 2026-08-02: WIN vs paper**
   (F1 0.7499, R@10 0.1580). See [BEST_RESULTS_kg_transformer.md](BEST_RESULTS_kg_transformer.md).

**Wave 2 unlocked** with `--kgtr_pool raw --kgtr_token rel_ej`.

### Wave 2 — transfer (copy-paste)

```powershell
cd D:\projects\current-git-project\LGKAT-Project\src
conda activate lgkat

# book (proposed book gate_es + kgtr raw)
python main.py --dataset book --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 0.0002 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion gate --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag thesis_kgtr_book_pool_raw_gate_es

# movie (proposed movie fixed bs8k + kgtr raw)
python main.py --dataset movie --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 8192 --l2_weight 5e-6 --lr 0.0025 --layer_size [32] --adj_type plain --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 1 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 8 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag thesis_kgtr_movie_pool_raw_fixed_bs8k

# restaurant (proposed restaurant + kgtr raw, H=2)
python main.py --dataset restaurant --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 32768 --l2_weight 1e-7 --lr 0.02 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att kg_transformer --n_heads 2 --kgtr_token rel_ej --kgtr_pool raw --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag thesis_kgtr_restaurant_pool_raw_fixed_bs32k
```

Confirm book/movie/restaurant flags against `BEST_RESULTS_proposed.md` (book: `lr 0.0002`, `early_stop 5`).

---

## 5. Scoreboards & file naming

| Family | Doc | Result pattern |
|--------|-----|----------------|
| Proposed champion | `BEST_RESULTS_proposed.md` | `*_product_proposed_*` |
| Multi-head product | `BEST_RESULTS_product_mh.md` | `*_product_mh_thesis_pmh_*` |
| KG Transformer | `BEST_RESULTS_kg_transformer.md` | `*_kg_transformer_thesis_kgtr_*` |

Never hand-edit result files; commit them with the code that produced them.

---

## 6. Phase 4 — improvement backlog (if v1 is weak)

Apply **one change at a time**, new `--exp_tag`, keep v1 as ablation baseline.

| Priority | Change | Why |
|----------|--------|-----|
| P1 | Token = `concat(u, r⊙ei, ej)` | **Tried — REJECTED** (`thesis_kgtr_v2_token_u_es25`: F1↓ vs v1) |
| **P2** | **`--kgtr_pool raw`:** QKV → weights; mix raw `e_j` | **Code landed** — smoke `thesis_kgtr_v2_pool_raw_es25` |
| P2b | Add FFN after LN | Defer — more params after P1 already hurt |
| P3 | Attn dropout 0.1 | Regularize if raw pool still unstable |
| P4 | `n_heads` sweep | Later |
| **Pivot** | Champion thesis arch = `product_mh`; kgtr = negative result | If P2 fails — still fills multi-head gap |

**P1 smoke (done — do not re-run as preferred):**
`--kgtr_token product --exp_tag thesis_kgtr_v2_token_u_es25`

**P2 smoke (pending approval):**

```powershell
cd D:\projects\current-git-project\LGKAT-Project\src
conda activate lgkat

python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion gate --use_cl 1 --cl_lambda 0.02 --cl_tau 0.2 --use_simgcl 1 --simgcl_lambda 0.02 --simgcl_eps 0.05 --lr_decay 0.98 --early_stop 12 --early_stop_min_epoch 25 --fast_eval 1 --exp_tag thesis_kgtr_v2_pool_raw_es25
```

Stop restaurant-only early-stop chases that already failed under `product_mh` unless
architecture changed.

---

## 7. Thesis narrative (draft)

1. Baseline: Product LGKAT (paper).
2. Proposed stack: fusion / CL / SimGCL ([BEST_RESULTS_proposed](BEST_RESULTS_proposed.md)).
3. Gap: Transformer / multi-head KG attention.
4. Intermediate: `product_mh` (multi-head product scoring).
5. Candidate: `kg_transformer` (personalized QKV over neighbors).
6. Ablations: product vs product_mh vs kg_transformer ± stack.
7. Honest protocol caveats (`topk_user_num`, hardware batch caps).

---

## 8. Pointer index

| Need | Where |
|------|--------|
| v1 design (approved) | `docs/superpowers/specs/2026-07-31-kg-transformer-design.md` |
| Implementation tasks | `docs/superpowers/plans/2026-07-31-kg-transformer.md` |
| NotebookLM notes | `docs/research_notes_notebooklm_transformer.md` |
| product_mh results | `docs/BEST_RESULTS_product_mh.md` |
| kg_transformer results | `docs/BEST_RESULTS_kg_transformer.md` |
| Compare to paper | `python scripts/compare_run_to_paper.py result/….txt <dataset>` |
| Paper cmds | `src/cmd.sh` |
| Proposed cmds | `src/cmd_proposed.sh` |
