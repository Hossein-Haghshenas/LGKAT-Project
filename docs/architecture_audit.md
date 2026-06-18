# OL-LGKAT — Phase 0 Architecture Audit

_Generated during Phase 0 (audit & reproduce). Describes the code as it exists in the
repository on 2026-06-18. Line references use the **current checked-out branch**
(`model-with-online-learning` @ `eac43c6`) unless stated otherwise._

---

## 1. Repository & branch landscape (important)

The thesis features are **scattered across ~14 branches**, not all on one line. The single
most relevant fact for this project:

| Branch | Tip | What it contains |
|---|---|---|
| `model-with-online-learning` (checked out) | `eac43c6` | **Bare base**: LightGCN + KGCN + product attention + fixed-α fusion. No NDCG, no gate, no ablation, no online code. |
| `major-version-1` | `d77f43c` | **Consolidated line**: everything in the base **plus** `--report_ndcg` (NDCG@K), `--fusion gate` (adaptive fusion gate), flag-gated contrastive loss, and `scripts/` for ablation, sparsity/cold-start, sensitivity sweep, significance testing, attention case-study export. |
| others (`base-model*`, `improved-base-model*`, `with-*`, `graph-*`, `feat/mokar-nsga2`, `dev`) | various | Experimental variants (PCA, diffusion, multi-head/transformer attention, GAT/GraphSAGE aggregators, NSGA-II HPO, contrastive). Not the build target. |

**Consequence:** Several deliverables the OL-LGKAT spec lists as "missing" already exist on
`major-version-1` (see §8). The branch we build OL-LGKAT on must be decided before Phase 1.

---

## 2. Data format & flow

Standard KGCN/RippleNet CTR format, one folder per dataset under `data/<name>/` (the
`data/` dir is **git-ignored**):

- `ratings_final.txt/.npy` — `user_id  item_id  label(0/1)`. **3 columns, no timestamp.**
  (verified: `data/music/ratings_final.txt` → `0\t32\t1`).
- `kg_final.txt/.npy` — `head  relation  tail`.
- `item_index2entity_id.txt` — item→KG-entity map (used by `preprocess.py`).

**Loading pipeline** ([src/data_loader.py](../src/data_loader.py)):
`load_data` → `load_rating` (reads ratings, caches `.npy`, splits, builds UI adjacency) +
`load_kg` (reads KG, builds undirected adjacency, fixed-size neighbor sampling per entity).

- **Split** — `dataset_split` ([data_loader.py:135](../src/data_loader.py)): **random** 6:2:2
  train/eval/test via `np.random.choice` (seeded by `args.seed`). **No temporal order.**
- **UI adjacency** — built from train positives only; normalized (`norm`/`plain`/`mean`);
  then raised to the power `--smoothing_steps` (`A^S`, paper Table 1) via `adjacency_power`
  ([data_loader.py:53](../src/data_loader.py)).
- **KG adjacency** — `construct_adj` samples a fixed `neighbor_sample_size` neighbors per
  entity into `adj_entity` / `adj_relation` arrays.

---

## 3. Model architecture ([src/model.py](../src/model.py), class `KGCN`)

TF1-style **graph mode** (`tf.compat.v1`, `disable_eager_execution()`), built once per run.
Two branches feeding a per-item fusion, scored by dot product.

```
                 user_indices                         item_indices
                      │                                     │
        ┌─────────────┴───────────┐            ┌────────────┴─────────────┐
        │  LightGCN on UI graph    │            │  KGCN on knowledge graph │
        │  _create_lightgcn_embed  │            │  get_neighbors→aggregate │
        │  (model.py:433)          │            │  (model.py:516,527)      │
        └───────┬──────────┬───────┘            └───────────┬──────────────┘
       user_emb_final   item_emb_cf                     item_emb_kg
                │           └──────────┐        ┌────────────┘
                │                ╔══════▼════════▼══════╗
                │                ║  FUSION (model.py:   ║   ← α blend, agg_type=weighted_avg
                │                ║  172–192)            ║      item_final = α·cf + (1−α)·kg
                │                ╚══════════╤═══════════╝
                │                           │ item_emb_final
                └────────────► dot ◄────────┘
                                │
                       score = Σ(user_final · item_final)   (model.py:198)
                       loss  = sigmoid-BCE + l2             (model.py:574–617)
```

- **LightGCN branch** (`_create_lightgcn_embed`, [model.py:433](../src/model.py)): stacks
  `n_layers` propagations over the normalized UI adjacency, mean-pools layer outputs, splits
  into user (`ua_embeddings`) and item (`ia_embeddings`) embeddings. (NGCF variant also present.)
- **KGCN branch** (`get_neighbors` + `aggregate`, [model.py:516](../src/model.py)): gathers
  multi-hop KG neighbors, aggregates with a personalized attention (see §4), `add_n` across hops.
- **Score & loss**: dot product → sigmoid → `sigmoid_cross_entropy_with_logits` + L2.

### Two distinct "attention/gate" mechanisms (do not conflate)

1. **Personalized knowledge-aware attention** — in
   [src/aggregators.py](../src/aggregators.py), `_mix_neighbor_vectors` (line 43). Weights the
   KG *neighbors* of an item by `softmax` of an interaction (e.g. `--att product`, paper Eq. 7,
   [aggregators.py:122](../src/aggregators.py)). This is the paper's **attention sub-network**.
2. **Fusion coefficient α** — [model.py:172–192](../src/model.py). Combines the **CF item
   embedding and the KG item embedding** into one. Currently a **fixed scalar hyperparameter**
   (`--alpha`, [main.py:110](../src/main.py)). **This is the OL-LGKAT Phase 4 target** (the
   "Forgetting-Aware Fusion Gate"), *not* the neighbor attention.

---

## 4. Training & evaluation ([src/train_n_runs.py](../src/train_n_runs.py))

- `train_n_runs` loops `--runs` times; each run resets the graph, rebuilds the model, trains
  `--n_epochs`, picks the best epoch by **eval AUC** (for CTR) and **eval Recall@10** (for top-K),
  reports the corresponding **test** metrics, then aggregates mean±std across runs.
- **CTR metrics** — `ctr_eval` / `model.eval` ([train_n_runs.py:328](../src/train_n_runs.py),
  [model.py:622](../src/model.py)): **AUC, F1**.
- **Top-K metrics** — `topk_eval` ([train_n_runs.py:340](../src/train_n_runs.py)): for sampled
  users, score all non-train items, sort, compute **Precision@K, Recall@K**. `--topk_user_num`
  (default 100; `0`=all users), `--topk_k_list` (e.g. `[2,10,50,100]`).
- **Checkpointing** — a `tf.train.Saver` is created ([train_n_runs.py:54](../src/train_n_runs.py))
  but in `--logging save` mode the session is **always initialized from scratch**
  ([train_n_runs.py:82–90](../src/train_n_runs.py)); there is **no resume**. Embeddings are
  dumped to `output/` at the best epoch only.
- **Outputs** — appended to `result/<model>_<dataset>_{ctr,topk,train_loss}_<att>.txt`, plus a
  `paper_style` mean±std block. `result/` **is git-tracked**; `output/` and `Pretrain/` are not.

---

## 5. Entry point ([src/main.py](../src/main.py)) — quality notes

- Global `np.random.seed(555)`; `--seed` default 555 (matches the project's reproduction seed).
- Large commented-out arg blocks for movie/music; the **active default dataset is `book`**.
- `--att` and `--smoothing_steps` are each declared **twice** (the second definition wins —
  argparse would otherwise error). Harmless but worth cleaning when we touch `main.py`.

---

## 6. `major-version-1` extensions (the likely build base for OL-LGKAT)

Confirmed by reading that branch:

- **NDCG@K** — `--report_ndcg` ([main.py:127]); `topk_eval` returns `ndcg`; logged in the
  `paper_style` block. (Phase 1 is largely done here, for the K in `--topk_k_list`.)
- **Adaptive fusion gate** — `--fusion {fixed,gate}` ([main.py:146]); when `gate`, the fixed α
  is replaced by a **per-item learned scalar** `g = sigmoid(W_g·[e_cf; e_kg] + b_g)`
  (`model.py:185–195`), **identity-initialized** (`W_g=0`, `b_g=logit(α)`) so step 0 equals the
  fixed-α baseline and adapts from there (`_init_weights_gate`, `model.py:336–353`).
  **→ This is the exact object Phase 4 extends with a historical-stability signal.**
- **Contrastive loss (InfoNCE)** — `--cl_lambda` ([main.py:135]), `_create_contrastive_loss`
  (`model.py:629`), off by default. _Spec says this module is unstable — do not enable it._
- **Scripts** — `scripts/ablation_table.py`, `scripts/sparsity_table.py` (cold-start buckets),
  `scripts/sensitivity_sweep.py`, `scripts/significance_test.py`, plus attention case-study
  export (`export_attention_case_study`) for interpretability. (Phase 6 is partly done here.)

---

## 7. Where each OL-LGKAT phase hooks in

| Phase | Hook point |
|---|---|
| 1 — NDCG@K | `topk_eval` ([train_n_runs.py:340]); already implemented on `major-version-1` (extend to all K + unit test). |
| 2 — Datasets | `preprocess.py` + new converters; loader reads `data/<name>/`. |
| 3 — Temporal split | replace `dataset_split` ([data_loader.py:135]) with timestamp-windowed split (needs a timestamp column — see §9). |
| 4 — Forgetting-aware gate | extend the gate at [model.py:185] / `_init_weights_gate` on `major-version-1`; add per-window checkpoint/resume to `train_n_runs`. |
| 5 — Baselines | new incremental training loop (fine-tune / replay / EWC / full-retrain) around the per-window step. |
| 6 — Ablation/cold-start/sensitivity/interpretability | mostly present in `scripts/` on `major-version-1`. |
| 7 — Aggregation | new script over `result/`. |

---

## 8. Spec "known gaps" vs. reality

| Spec claim | Reality |
|---|---|
| "NDCG is not implemented anywhere" | True on the checked-out branch; **already implemented on `major-version-1`** (NDCG@K). |
| "data split is random (no timestamps)" | **True everywhere** — and the existing datasets have **no timestamp column** at all (§9). |
| "contrastive module is unstable (do NOT use)" | Module exists on `major-version-1`, flag-gated, **off by default** — consistent with the instruction to leave it off. |
| "full ablation and a cold-start experiment are missing" | **Already present on `major-version-1`** (`ablation_table.py`, `sparsity_table.py`). |

---

## 9. Risks / open issues

1. **No timestamps in existing data.** `ratings_final` is `user item label`. The temporal/
   streaming protocol (Phase 3) — the backbone of the "online" claim — **cannot** be run on the
   current music/book/restaurant/movie data without recovering timestamps from raw sources.
   ML-1M and Yelp2018 (which carry timestamps) become essential, not optional.
2. **Branch decision blocks downstream work.** Building on the bare branch would re-implement
   NDCG, the gate, ablation, and cold-start that already exist on `major-version-1`. The spec's
   "current state" matches the bare branch, but the project memory and the code both point to
   `major-version-1` as the real foundation. **Needs user confirmation (see Phase 0 report).**
3. **TF1 graph mode + full rebuild per run.** No native resume; Phase 4 must add explicit
   save/restore of weights + optimizer slots + window index + stability stats.
4. **`result/` is tracked**, so every run mutates tracked files (append/overwrite). Smoke-test
   outputs are restored via `git checkout -- result/`. Future runs should write to per-experiment
   subfolders to keep the tree clean.
5. **`main.py` hygiene** — duplicate args, hardcoded default dataset — clean up opportunistically.

---

## 10. Environment (verified)

- conda env **`lgkat`**: Python **3.9.25**, TensorFlow **2.9.1**, **GPU:0 visible**. ✅
- Run from `src/` (paths are relative: `../data`, `../result`, `../output`, `../Pretrain`).
- Hardware budget: 16 GB RAM / 16 GB GPU → short, chunked runs only; long runs handed off.
