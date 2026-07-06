# LGKAT — LightGCN + knowledge-aware attention recommender

TensorFlow implementation of the Scientific Reports paper "A novel recommender system
using light graph convolutional network and personalized knowledge-aware attention
sub-network" (built on the COAT/KGCN codebase). Current goal: architectural improvements
(attention, contrastive learning, fusion…) that measurably beat the paper's Table 2/3
numbers — feeding Hossein's MSc thesis (see the Thesis Project folder's CLAUDE.md) and
a follow-up paper. Pristine upstream copy: D:\projects\LGKAT-main — never modify it;
diff against it to see what we changed.

## Stack & environment
- Python 3.9, TensorFlow 2.9.1 **in TF1 graph mode**: `tf.compat.v1`, placeholders,
  sessions, `variable_scope`/`get_variable`. Eager execution is disabled in model.py —
  never introduce Keras layers/`Model.fit` or eager idioms; new ops must build into the
  existing static graph.
- Conda env: `lgkat` (GPU).

## Layout
- `src/` — model.py (`KGCN` class = the whole model), aggregators.py, train_n_runs.py
  (train/eval loop + result writing), main.py (CLI), data_loader.py, preprocess.py,
  cmd.sh (canonical paper-reproduction command per dataset).
- `scripts/` — post-hoc analysis of result files. Paper targets live in
  scripts/paper_targets_product.py.
- `data/<dataset>/` — gitignored, obtained from the corresponding author. Never modify or
  regenerate `*_final.*` without asking: preprocess.py resamples negatives, silently
  breaking comparability with every recorded result.
- Dataset codes: `music` = Last.fm, `book` = Book-Crossing, `restaurant` = Dianping,
  `movie`/`ml1m` = MovieLens variants — confirm which variant a result file came from
  before quoting it.
- `result/` — **tracked** experiment record. Never hand-edit; commit result files together
  with the code that produced them.
- `output/`, `Pretrain/` — generated embeddings & checkpoints.
- `configs/`, `notebooks/`, `logs/`, `embeddings/` — empty or written by other branches'
  code; ignore on this branch.

## Running
- NEVER launch training yourself — Hossein runs all training on his GPU (hours-long).
  Write exact copy-paste commands for him, then analyze the result files he produces.
  For quick checks suggest `--dataset music` (smallest); `restaurant` with
  `--topk_user_num 0` is extremely slow.
- Training runs from `src/` (`cd src && python main.py …`): all I/O paths are relative
  (`../data/`, `../result/`, `../output/`), so from the repo root it breaks. The comment
  in cmd.sh saying repo root works is wrong.
- Analysis scripts run from the repo root:
  `python scripts/compare_run_to_paper.py result/KGCN_LightGCN_book_product.txt book`.
- Always pass the full flag set from cmd.sh — main.py's argparse defaults are NOT the
  paper settings (a per-dataset block is merely commented in/out; leave those comment
  blocks alone).

## Git
- Full freedom: commit (feat:/fix: prefix, imperative), push, switch branches.
- Experiments live on dedicated branches (`with-<idea>`, `graph-…`); `major-version-1`
  is the current mainline.
- Commit or stash dirty `result/*.txt` before any checkout — losing run logs means
  re-running hours of training.

## Reproducibility — the rule that matters most
- Every new feature goes behind a CLI flag that is OFF by default; a default-flag run must
  reproduce previous results and result-file text **byte-for-byte** (follow the existing
  pattern: `--report_ndcg`, `--use_cl`, `--fusion`, `--exp_tag`).
- Never change: seeds (`np.random.seed(555)`, `args.seed`), default hyperparameters in
  main.py, or the format/naming of existing result files. Route new variants to new
  filenames via `--exp_tag`.
- List-valued CLI args are parsed with `eval()` — pass quoted Python literals
  (`--topk_k_list '[2,10,50,100]'`) and keep that pattern for new list args.

## Claiming an improvement
- Paper numbers are mean±std over `--runs 3`; a single run is never evidence. Compare with
  scripts/compare_run_to_paper.py; for close calls use scripts/significance_test.py
  (paired per-run tests). Default `--topk_user_num 100` is fast but higher-variance than
  the paper's all-users protocol (`0`).
- Exclude crashed runs before averaging — one contaminated a Last.fm mean before.
