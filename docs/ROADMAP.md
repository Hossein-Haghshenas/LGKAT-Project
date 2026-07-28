# LGKAT — Thesis & code roadmap

Living handoff for future chats/agents. Last updated: **2026-07-28**.

Operational rules (how to work in this repo) stay in [CLAUDE.md](../CLAUDE.md) /
[AGENTS.md](../AGENTS.md). Deep engineering analysis stays in
[KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md). This file is the **agreed path forward** after
aligning Hossein’s MSc proposal with the current codebase and best results.

---

## 1. Context in one paragraph

This repo implements **LGKAT** (Scientific Reports paper: LightGCN + personalized
knowledge-aware attention). Active evidence branch: **`major-version-1`**. Goal: beat the
paper’s Table 2/3 Product numbers with flag-gated architectural upgrades, feed Hossein’s
thesis, and support a follow-up article. Pristine upstream (never modify):
`D:\projects\LGKAT-main`.

Hossein’s proposal (v5) title promises a **knowledge-aware recommender integrating
Transformer networks and self-supervised learning**, built on the LGKAT paper as
reference [1], same public datasets (Last.fm / Book-Crossing / Dianping), Python +
TensorFlow, ~6 months.

---

## 2. Decisions already made (do not reopen casually)

1. **Current proposed model beats the paper on mean metrics** for music, movie, and
   restaurant; book is **MIXED** (F1 up, R@10 slightly down). Full tables and copy-paste
   recipes: [BEST_RESULTS_proposed.md](BEST_RESULTS_proposed.md).
2. **Freeze `major-version-1` as the evidence track.** Do not land experimental thesis
   features on it in a way that risks default-flag reproduction. Treat it as the champion
   baseline for all later comparisons.
3. **Two-track plan:** keep this branch untouched for claims; audit other experiment
   branches for salvageable ideas; then create a **new clean branch from `major-version-1`**
   for thesis-gap work (Transformer / multi-head attention / related).
4. **If the new model does not beat the proposed champion**, that is fine: the thesis
   compares the new architecture to the current best (`BEST_RESULTS_proposed`) — a negative
   or mixed result is still a valid scientific outcome.
5. **Result scoreboards are per model family**, not one global file:
   - Current: `docs/BEST_RESULTS_proposed.md`
   - Future: `docs/BEST_RESULTS_<model_or_branch_tag>.md` (same structure: scoreboard,
     recipes, rejected variants, claim caveats).

---

## 3. Proposal ↔ code gap map

| Proposal claim | Status on `major-version-1` |
|----------------|-----------------------------|
| Beat / improve on LGKAT Product baseline | Largely done — see proposed scoreboard |
| Self-supervised / contrastive learning | Done behind flags: `--use_cl`, `--use_simgcl` |
| Adaptive fusion of CF + KG item views | Done: `--fusion gate` (identity init) |
| Cold-start / sparsity narrative | Eval tooling: `--sparsity_eval` (+ buckets); still need strong proposed runs reported per bucket for hypothesis text |
| Interpretability via attention | Done: attention export (Task 6) |
| **Transformer networks** (title + hypothesis 5) | **Main gap.** `with-transformer` branch is stalled (no real Transformer in model). Closest salvage: `with-multi-head-attention` |
| Unified UI+KG graph (KGAT-style) | Not LGKAT design; two views meet at fusion + CL only — do not rewrite unless thesis text requires it |

**Thesis risk:** defending the English title literally without a Transformer (or a clearly
named Transformer-style multi-head attention module) will look inconsistent. Either
implement a flag-gated module on the new branch, or align proposal wording with what we
actually ship.

---

## 4. What “proposed” already includes (champion recipes)

Per-dataset best `exp_tag`s and commands are canonical in
[BEST_RESULTS_proposed.md](BEST_RESULTS_proposed.md). Ingredients in short:

| Dataset | Best tag | Main extras vs paper |
|---------|----------|----------------------|
| music | `proposed_v3` | gate + CL + SimGCL + LR decay + early stop |
| book | `proposed_book_gate_es` | gate + higher lr + LR decay + early stop; **no** CL |
| movie | `proposed_movie_fixed_bs8k` | fixed fusion; batch 8192 + scaled lr (paper 65536 OOMs) |
| restaurant | `proposed_restaurant_fixed_bs32k` | fixed fusion; batch 32768 + lr 0.02 |

**Claim caveats (always when quoting numbers):**

- Means over `--runs 3`; use `scripts/significance_test.py` for close calls.
- Proposed runs used `--topk_user_num 100`; paper tables use all eligible users (`0`).
- Movie/restaurant training batch/lr ≠ paper Table 1 (hardware).
- Never regenerate `data/*/ratings_final.*` / `kg_final.*`.

Training is run by Hossein on GPU — agents write copy-paste commands only (`cd src &&
python main.py …`). Never launch long training from tooling.

---

## 5. Agreed next steps (ordered)

### Step A — Freeze evidence (now)

- [x] Rename scoreboard to `docs/BEST_RESULTS_proposed.md` (distinct from future models).
- [ ] Commit docs handoff (`ROADMAP.md`, rename) when the user asks.
- [ ] Do not push experimental commits onto `major-version-1` default path.

### Step B — Audit other branches (read-only salvage)

Compare each branch to `major-version-1`; note one idea, merge-base age, and whether the
idea is already mainlined. Priority for thesis-gap filling:

| Priority | Branch | Salvage for |
|----------|--------|-------------|
| High | `with-multi-head-attention` | Transformer-adjacent novelty (`--att product_mh` style) |
| High | `feat/mokar-nsga2` | KG relation-selection mask (orthogonal denoise) |
| Medium | `graph-attention-networks` | Ablation / GAT baseline narrative |
| Medium | `graph-sage-architecture` | Ablation baseline |
| Low | `with-transformer` | Title alignment only if rewritten from scratch on new branch |
| Low | `base-model-with-cl` | Mostly superseded (LR decay / ES already on mainline) |
| Separate track | `model-with-online-learning` | Temporal/online thesis chapter — not the current gap |

Full branch atlas: [KNOWLEDGE_BASE.md §3](KNOWLEDGE_BASE.md).

### Step C — New clean branch from `major-version-1`

Suggested name (pick one when creating): `thesis-transformer` or
`with-product-mh` / `with-thesis-ssl-transformer`.

Rules on that branch (same as repo contract):

- Every new feature behind a CLI flag, **OFF by default**.
- Default-flag run must still match previous result-file text byte-for-byte.
- Route new runs with `--exp_tag` (never overwrite proposed champion files).
- New scoreboard: `docs/BEST_RESULTS_<tag>.md`.
- Compare always against [BEST_RESULTS_proposed.md](BEST_RESULTS_proposed.md) under the
  **same protocol** (same `topk_user_num`, runs, dataset flags from `cmd.sh` / proposed
  recipes).

### Step D — Implementation focus for the new branch

Fill the proposal gap first; accuracy gain is secondary:

1. **Transformer-style KG attention** (preferred incremental path: multi-head product
   attention salvaged/cleaned from `with-multi-head-attention`, flag-gated).
2. Optionally **MOKAR relation mask** as KG denoise (independent ablation).
3. Keep proposed training tricks available (`--fusion`, `--use_cl`, `--use_simgcl`,
   `--lr_decay`, `--early_stop`, `--fast_eval`) so the new module is tested both alone and
   stacked on the champion recipe.
4. First smoke dataset: **music**; then book/movie; restaurant last (slow / heavy).

### Step E — Thesis write-up regardless of win/lose

- Champion = proposed model numbers + recipes.
- Candidate = new-branch model.
- Report mean±std, paired tests where close, protocol caveats, and ablations
  (attention-only vs +CL vs full stack).
- Hypotheses from the proposal map to experiments: KG value, sparsity/cold-start buckets,
  contrastive, Transformer/multi-head vs simple propagation, interpretability cases.

---

## 6. What not to do

- Do not treat a single run as evidence.
- Do not “normalize” per-dataset paper flags (α, S, dim, lr) across datasets.
- Do not modify `D:\projects\LGKAT-main`.
- Do not regenerate `*_final` data files without an explicit human decision.
- Do not merge stale experiment branches wholesale into `major-version-1` — cherry-pick
  ideas onto the new thesis branch instead.
- Do not start LGKAT-v2 (PyTorch) until the evidence-track thesis story is locked; v2 remains
  a longer-term track in KNOWLEDGE_BASE §13.

---

## 7. Quick pointer index

| Need | Where |
|------|--------|
| Repo work rules | `CLAUDE.md` / `AGENTS.md` |
| Architecture, bottlenecks, branch atlas | `docs/KNOWLEDGE_BASE.md` |
| Champion numbers + commands | `docs/BEST_RESULTS_proposed.md` |
| Paper target constants | `scripts/paper_targets_product.py` |
| Compare a result file to paper | `python scripts/compare_run_to_paper.py result/….txt <dataset>` |
| Paired significance | `scripts/significance_test.py` |
| Paper reproduction cmds | `src/cmd.sh` |
| Proposed-style cmds | `src/cmd_proposed.sh` + scoreboard recipes |

---

## 8. Status snapshot

| Item | State |
|------|--------|
| Evidence branch | `major-version-1` (freeze for experiments) |
| Champion scoreboard | `docs/BEST_RESULTS_proposed.md` |
| Thesis/code gap | Transformer / multi-head attention not on mainline |
| Next engineering move | Branch audit → new branch from tip → flag-gated attention module |
| Success criterion for new model | Prefer beating proposed; acceptable to lose if comparison is rigorous |
