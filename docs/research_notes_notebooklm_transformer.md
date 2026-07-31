# Research notes — NotebookLM + Phase-1 audit (temporary)

**Date:** 2026-07-29  
**Branch context:** `thesis-transformer` (from `major-version-1` tip); evidence track untouched.  
**Notebook:** [LGKAT Gemini NotebookLM](https://notebooklm.google.com/notebook/0ed9e610-91d0-48db-94e2-fa85996de492) (34 sources).  
**Status:** Temporary research handoff for Phase-2 design approval — **not** an experiment result scoreboard.

Access note: Cursor’s default Chrome DevTools MCP launches a separate automation browser that Google rejects for sign-in. Session used Option B (`--remote-debugging-port=9222`); queries were sent over CDP to the logged-in NotebookLM tab. `~/.cursor/mcp.json` was updated to pass `--browserUrl http://127.0.0.1:9222` for future reconnects (reload MCP after Chrome debug is up).

---

## 1. Phase-1 audit (recap)

| Branch | Verdict |
|--------|---------|
| `with-multi-head-attention` | Salvage **idea** only (multi-head over KG neighbors). Do **not** merge: ON-by-default, replaces ConcatAggregator, predates `--att product` / fusion / CL / interpret, stale QKV+LN+residual block. |
| `feat/mokar-nsga2` | Schema only for later: `relation_mask` in `construct_kg` / `load_data`; orthogonal KG denoise. |
| `with-transformer` | **Stalled** — only `dataset_logger.ipynb`; no Transformer in model. |

Proposed incremental design from audit + `docs/ROADMAP.md` / `KNOWLEDGE_BASE.md` P2:

- **`--att product_mh`**: multi-head variant of paper Product attention inside `_mix_neighbor_vectors`, keep ConcatAggregator.
- Flags off by default; `--n_heads`; new `--exp_tag`; never overwrite `BEST_RESULTS_proposed` files.

---

## 2. Questions asked to NotebookLM

### Q1 (broad roadmap)

Topics: KG-aware LightGCN/KGCN attention; Graph Transformer over KG neighbors; multi-head for product-style `concat(u, r⊙ei, ej)`; cross-view CL / SimGCL; cold-start; low-risk upgrades on dual-view LGKAT without unified UI+KG graph. Structured as A/B/C (feasible / out of scope / recommended).

### Q2 (implementation fork)

- **MH-A:** multi-head *product scoring* (`--att product_mh`) — head-specific `W_h` on same concat features → one softmax → weighted neighbor sum.  
- **MH-B:** full QKV self-attention over fixed neighbor set T (Transformer block, residual+LN).  
Ask: which first for beating Product LGKAT + thesis title; default `n_heads` for `dim ∈ {8,32,64}`.

---

## 3. NotebookLM answers (condensed)

### From Q1

**Attention designs (from LGKAT paper sources in notebook):** Product and Bi-Perceptron were strongest among tested attention functions; Bi-Interaction and classic KGCN-style also discussed. Product = element-wise `r⊙ei` then concat with `u` and `ej` (matches our mainline `--att product`).

**Graph Transformer angle:** Treat fixed sampled neighbors as a short sequence; replace simple weighted sum with self-attention; keep relation-aware structure (not NLP positional encodings alone). Compatible with fixed `T` (paper neighbor sample sizes).

**MHA pitfalls:** Over-parameterization on sparse Book-Crossing; cost on large Restaurant; need regularization / careful LR.

**CL / SSL:** Dual CF+KG views suit cross-view InfoNCE; XSimGCL-style noise CL called out as simple stack; aligns with flags we already have (`--use_cl`, `--use_simgcl`).

**Cold-start:** KG side info, LightGCN’s simplified propagation, α / fusion between UI and KG — already in proposed recipes + `--sparsity_eval` tooling.

**A — Feasible on TF1 + codebase:** Product/Bi-Perceptron family; cross-view / SimGCL-style CL; per-dataset LR/L2 (already in recipes).  
**B — Out of scope now:** MoE/Pathways, LLM text pretrain for entities, ModernBERT long-context local/global attention.  
**C — NotebookLM’s recommended thesis arc:** “Multi-Head Personalized KG Transformer” = MHA over neighborhoods + relation-aware multi-head product scores + keep fusion + add/keep contrastive alignment.

### From Q2

NotebookLM’s **clear pick: MH-B first** (full QKV self-attention).

Reasons given: stronger “Graph Transformer” thesis wording; models **intra-neighbor** dependencies (Product scores each neighbor independently); fixed `T` fits TF1 static shapes; residual+LN as in Transformer blocks; suggested defaults `n_heads=4` (dim 32/64), `2–4` for dim 8.

---

## 4. Synthesis with Phase-1 / repo contract

| Criterion | MH-A `product_mh` | MH-B full QKV |
|-----------|-------------------|---------------|
| Thesis “Transformer / multi-head” wording | Adequate (multi-head KG attention; Transformer-**style**) | Stronger literal “Graph Transformer” |
| Continuity with paper Product / champion stack | High — same score features, ConcatAggregator | Low — different aggregation semantics; stale branch already tried this shape |
| Reproducibility / flags rule | Easy — new `att` branch only when selected | Harder — new weights, residual, LN; must still keep default path untouched |
| Risk vs proposed champion (gate/CL/SimGCL) | Lower — can stack on `proposed_v3` flags | Higher overparam (Book) + harder ablations |
| Salvage hygiene | Matches ROADMAP / KB P2 | Tempting to copy stale `MultiHeadAttentionAggregator` (reject) |

**Joint recommendation for Phase-3 (design to confirm):**

1. **Implement MH-A first** as `--att product_mh` + `--n_heads` (default 4 when used; assert `dim % n_heads == 0`).  
   - Rationale: fills proposal gap with minimum protocol risk; matches `KNOWLEDGE_BASE` iteration 5 / ROADMAP Step D; keeps fusion/CL/SimGCL/early-stop/`fast_eval` composable.  
   - Thesis language: “personalized multi-head knowledge-aware attention (Transformer-style multi-head scoring over KG neighbors).”

2. **Treat MH-B as optional Phase-3b / later ablation** (`--att kg_transformer` or similar), rewritten clean (not merge of `with-multi-head-attention`), only if MH-A fails to beat **paper** Product or narrative review demands literal self-attention.

3. **Success bar (unchanged):** mean±std over ≥3 runs; prefer beating proposed; **acceptable** if beats paper Product under same protocol even if below `BEST_RESULTS_proposed`. Compare under same `topk_user_num` / recipe family.

4. **Defer MOKAR relation mask** until after `product_mh` smoke on music.

5. **Out of scope (agree with NotebookLM B):** MoE, LLM entity init, unified UI+KG rewrite, LGKAT-v2 PyTorch until evidence story locked.

---

## 5. Concrete Phase-3 sketch (pending your approval)

**Files:** `src/aggregators.py`, `src/model.py`, `src/main.py` only (plus later `docs/BEST_RESULTS_product_mh.md` when runs exist).

**Mechanism (MH-A):**

- Features per neighbor unchanged: `z = concat(u, r⊙ei, ej)` ∈ R^{3d}.  
- Heads `h=1..H`: `score_h = leaky_relu(z W_h + b_h)` (or shared bias pattern matching product).  
- Combine heads (mean logits or concat→linear→scalar), softmax over neighbors, weighted sum → existing ConcatAggregator path.  
- Export `attention_scores` as mean-over-heads (Task 6 shape preserved).  
- Create att weights **only** if `att == product_mh`.

**First smoke command family:** music near `proposed_v3` but `--att product_mh --n_heads 4 --exp_tag thesis_pmh_v1` (and a pure ablation without CL later).

**Do not** overwrite `result/*proposed*` or change default `--att`.

---

## 6. Caveats

- NotebookLM can be inaccurate; numbers/citations in its prose were not re-verified against PDFs here.  
- Its MH-B preference is noted; **ordering MH-A first is an engineering/reproducibility override**, not a claim that MH-B is scientifically useless.  
- Protocol caveats from `BEST_RESULTS_proposed.md` still apply (`topk_user_num 100`, batch/lr hardware deviations on movie/restaurant).

---

## 7. Deeper design talk with NotebookLM (2026-07-29, later session)

We described our **working dual-view LGKAT** + constraints + concrete **MH-A `product_mh`** plan and asked for critique (not a push back to full QKV unless fatal).

### NotebookLM critique of MH-A

**Strengths:** multi-perspective scoring on dense `(u,r,ei,ej)` features; more stable semantic signal into fusion; low risk vs full QKV while keeping fixed `T` / ConcatAggregator.

**Risks:** overparam on Restaurant (`d=8`); head collapse if heads see identical features; weak gradients / dead heads on sparse Book.

**Recommendations accepted into draft v1 spec:**

| Choice | Spec |
|--------|------|
| Combine | **Mean of logits** (not concat+project for v1) |
| Features | **Channel split (Option A):** `z_h = concat(u_h, r_h⊙ei_h, ej_h)` ∈ R^{3·head_dim} |
| Scale | Per-head: `scaled_h = score_h / sqrt(head_dim)`, then mean over heads, then softmax |
| `H` | `d=8→2`, `d=32→4`, `d=64→4` (try 8 later on book) |
| Dropout on att | **0** for first smoke |
| Variables | Create `W_h` **only** when `att==product_mh` |
| Title wording | Prefer “Multi-Head Knowledge-Aware Attention”; reserve literal “Transformer” unless LN+residual self-attn added later |

**Music smoke checklist (NotebookLM):**

1. `--att product` paper-style sanity (default path / paper flags).  
2. `--att product_mh --n_heads 4` alone (`mh_only`) — no CL.  
3. `product_mh` + proposed_v3 stack (`mh_stack`).

**Evidence it works (not luck):** Run2 R@10 ↑ and preferably lower std vs single-head; Run3 F1 beats `proposed_v3`; attention maps less spiky / more diverse neighbors.

### Engineering note (ours)

Channel-split implies **`dim % n_heads == 0`** (Restaurant: prefer `H=2`). This is a small change vs earlier Phase-1 note that scalar-per-head on full `z` did not require divisibility — split is preferred for thesis narrative and head diversity.

---

## 8. Implementation status (Phase 3 started 2026-07-29)

Landed on `thesis-transformer` (flag-gated):

- `src/aggregators.py`: `--att product_mh` channel-split product scoring + scale + mean logits
- `src/model.py`: per-head `att_w_mh_*` / `att_b_mh_*` only when `att==product_mh`; L2 included
- `src/main.py`: `--n_heads` (default 4; unused unless `product_mh`)

Future scoreboard after runs: `docs/BEST_RESULTS_product_mh.md` (do not overwrite proposed).

---

## 9. Waiting on

Hossein GPU smoke runs (commands in chat / below). Agents do not launch training.
