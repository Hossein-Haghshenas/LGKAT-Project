# OL-LGKAT — Phase 5: Continual-Learning Baselines

All baselines run under the **same temporal protocol** (Phase 3) so they are directly
comparable. They are selected by flags on the temporal path (`--split temporal`) — no separate
training scripts, so the model, data, eval, and per-window timing are identical across methods.

## Baseline → command mapping

| Method | Role | Distinguishing flags |
|---|---|---|
| **Full-Retrain** | quality upper bound (expensive) | `--protocol batch --fusion fixed` |
| **Fine-Tune** | lower bound (forgetting-prone) | `--protocol incremental --fusion fixed --baseline none` |
| **Experience Replay** | sample & replay the past | `--protocol incremental --fusion fixed --baseline replay --replay_ratio R` |
| **EWC** | regularization (anchor stable weights) | `--protocol incremental --fusion fixed --baseline ewc --ewc_lambda L` |
| **OL-LGKAT (ours)** | forgetting-aware fusion gate | `--protocol incremental --fusion forgetting_gate` |
| _(ablation)_ Adaptive gate | gate without stability signal | `--protocol incremental --fusion gate` |

All honor `--gpu_id -1` (CPU) and `--deterministic 1` (reproducible + exact resume).

## Quality vs. time / memory table

`scripts/baselines_table.py` reads every `result/temporal/*.json` and reports per method:
- **quality** — avg AUC, avg & final NDCG@10 across evaluation windows;
- **time** — total wall-clock + avg seconds/window (logged per window in each run);
- **memory proxy** — peak training rows/window: Full-Retrain grows with history, Replay ≈2×,
  Fine-Tune / gate ≈1×. EWC additionally stores a diagonal Fisher + anchor (~3× the embedding
  matrices).

### Demonstration (smoke-scale: ml1m, dim 8, 1 epoch, T=5, CPU, deterministic)

| Method | avg AUC | avg NDCG@10 | final NDCG@10 | total s | peak rows/win |
|---|---|---|---|---|---|
| Experience Replay | 0.5748 | 0.1227 | 0.1823 | 66.3 | 301,510 (≈2×) |
| OL-LGKAT (forgetting gate) | 0.5607 | 0.1082 | 0.1453 | 62.8 | 150,755 (1×) |
| Fine-Tune | 0.5601 | 0.1032 | 0.1527 | 63.1 | 150,755 (1×) |
| EWC (λ=1.0) | 0.5601 | 0.1032 | 0.1527 | 76.5 | 150,755 (1×) |
| Full-Retrain (batch) | 0.5536 | 0.1005 | 0.1595 | 66.7 | 603,020 (grows) |

**Caveats — these are smoke-scale numbers, not conclusions:**
- **EWC at λ=1.0 is identical to Fine-Tune** (to 4 decimals): the `F·Δθ²` penalty is ≪ the base
  loss at this scale, so λ=1.0 has no effect. EWC needs a much larger λ (typically 1e3–1e5) —
  swept in Phase 6.
- **Full-Retrain is not yet the upper bound** here: with only 1 epoch it underfits the growing
  cumulative set each window (its *final* NDCG is competitive though). Proper epochs are needed
  for the expected ordering.
- The **forgetting gate** is marginally above Fine-Tune on average but below Replay; at 1 epoch /
  dim 8 it has little room to learn the stability→fusion mapping. **Whether it genuinely helps is
  decided by the Phase 6 ablation** (proper config, runs=3), not by this smoke.

> Smoke-scale numbers validate the pipeline only — they are **not** thesis results (1 epoch,
> dim 8, 30 top-K users). Use the hand-off commands below for reportable numbers.

## Hand-off: the real comparison (heavy — run on your side)

For the thesis table, run each method at a realistic config (larger `--dim`, more `--n_epochs`,
more windows, all top-K users). Repeat each with seeds 555/556/557 for mean±std (Phase 6).
Example single-run commands (from `src/`, CPU):

```bash
C="--dataset ml1m --aggregator concat --neighbor_sample_size 4 --dim 32 --n_iter 1 \
   --batch_size 8192 --l2_weight 1e-5 --lr 2e-2 --layer_size [32] --adj_type norm \
   --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] \
   --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 1 --att product --gpu_id -1 \
   --topk_user_num 0 --topk_k_list [2,10,50,100] --split temporal --n_windows 10 \
   --n_epochs 10 --deterministic 1"

python main.py $C --protocol batch       --fusion fixed                              # Full-Retrain
python main.py $C --protocol incremental --fusion fixed                              # Fine-Tune
python main.py $C --protocol incremental --fusion fixed --baseline replay --replay_ratio 1.0
python main.py $C --protocol incremental --fusion fixed --baseline ewc    --ewc_lambda 1.0
python main.py $C --protocol incremental --fusion forgetting_gate                    # OL-LGKAT
python ../scripts/baselines_table.py                                                 # build table
```

(`--topk_user_num 0` = all overlapping users — slower; reduce for quicker iterations. EWC λ and
replay ratio are swept in Phase 6 sensitivity.)
