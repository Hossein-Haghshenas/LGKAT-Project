# KG Transformer (`--att kg_transformer`) Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox syntax.
> Parent roadmap: `docs/THESIS_KG_TRANSFORMER_ROADMAP.md`.
> Spec: `docs/superpowers/specs/2026-07-31-kg-transformer-design.md`.

**Goal:** Flag-gated personalized QKV self-attention + residual + LN over KG neighbors.

**Architecture:** New `att` branch in `_mix_neighbor_vectors`; weights only when selected;
ConcatAggregator unchanged.

**Tech Stack:** Python 3.9, TensorFlow 2.9.1 in TF1 graph mode (`tf.compat.v1`).

## Global Constraints

- New features OFF by default; default-flag runs byte-for-byte unchanged.
- Never change seeds, paper defaults in `main.py`, or existing result filenames.
- Do not launch training; give Hossein copy-paste commands from `src/`.
- Do not merge `with-multi-head-attention`.
- Do not modify `D:\projects\LGKAT-main` or regenerate `*_final` data.
- List CLI args stay `eval()`-parsed Python literals.

---

## File map

| File | Responsibility |
|------|----------------|
| `src/aggregators.py` | Forward: tokens → MHA → LN → mean pool; export attn |
| `src/model.py` | Create vars; assert heads; L2 |
| `src/main.py` | Document `kg_transformer` in help |
| `docs/BEST_RESULTS_kg_transformer.md` | Scoreboard stub → fill after runs |

---

## Task 1: Weights + wiring in `model.py`

- [x] Extend att allow-list to include `kg_transformer` (same places as `product_mh`).
- [x] Assert `dim % n_heads == 0` for `kg_transformer`.
- [x] In `_init_weights_att`, create only-if-selected vars (W_in, QKV/U/O, LN).
- [x] Add `kg_transformer` to L2 loop (`att_w*`).

## Task 2: Forward in `aggregators.py`

- [x] Add `if att == 'kg_transformer':` block with print.
- [x] Build tokens, MHA, LN, mean pool; early return; export attn.

## Task 3: CLI help in `main.py`

- [x] Mention `kg_transformer` in `--att` / `--n_heads` help.

## Task 4: Docs

- [x] Roadmap / spec / plan / scoreboard stub written.
- [ ] After first Hossein results, fill `docs/BEST_RESULTS_kg_transformer.md`.

## Task 5: Smoke commands (do not run training)

See `docs/THESIS_KG_TRANSFORMER_ROADMAP.md` Wave 1–2.

```powershell
cd D:\projects\current-git-project\LGKAT-Project\src
conda activate lgkat

python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att kg_transformer --n_heads 4 --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fast_eval 1 --exp_tag thesis_kgtr_mh_only

python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att kg_transformer --n_heads 4 --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion gate --use_cl 1 --cl_lambda 0.02 --cl_tau 0.2 --use_simgcl 1 --simgcl_lambda 0.02 --simgcl_eps 0.05 --lr_decay 0.98 --early_stop 12 --early_stop_min_epoch 25 --fast_eval 1 --exp_tag thesis_kgtr_v3stack_es25
```

---

## Done when

- [x] Default / `--att product` path unchanged in code review (flag-gated only).
- [ ] `--att kg_transformer` builds and trains (Hossein).
- [ ] Roadmap + scoreboard updated after Wave 1 numbers exist.
