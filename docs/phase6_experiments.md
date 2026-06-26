# OL-LGKAT — Phase 6: Ablation, Cold-Start, Sensitivity, Interpretability

All Phase 6 experiments run under the temporal protocol via flags + two small drivers
(`scripts/run_temporal_grid.py`, the per-run `--coldstart` / `--interpret` switches). Every
heavy run is meant to be repeated with seeds **555/556/557** for mean±std; the smoke tables
below use 2 seeds at dim 8 / 1 epoch only to validate the pipeline — **they are not results.**

## 1. Ablation — `scripts/run_temporal_grid.py --experiment ablation`

| cell | what it removes / changes |
|---|---|
| `full_fixed` | full model, fixed-α fusion |
| `full_gate` | full model, adaptive gate |
| `full_forgetting` | full model, **forgetting-aware gate (ours)** |
| `no_kg_lightgcn` | drop the KG branch (`--model_type LightGCN`) |
| `no_lightgcn_kgcn` | drop the LightGCN branch (`--model_type KGCN`) |

### Smoke demonstration (ml1m, dim 8, 1 epoch, T=4, 2 seeds)

| cell | avg AUC | avg NDCG@10 | final NDCG@10 |
|---|---|---|---|
| full_fixed | 0.5717 | 0.0931 ± 0.019 | 0.1295 |
| full_gate | 0.5715 | 0.0923 ± 0.032 | 0.1208 |
| full_forgetting | 0.5712 | 0.0930 ± 0.032 | 0.1216 |
| no_kg_lightgcn | 0.5696 | **0.0600** ± 0.010 | 0.0902 |
| no_lightgcn_kgcn | 0.5689 | 0.0845 ± 0.019 | 0.1516 |

**Reading (smoke):** removing **either branch hurts** average NDCG@10 (0.060 / 0.085 vs ~0.093)
— a clean architecture-ablation signal that both LightGCN and KG contribute. But `fixed` ≈ `gate`
≈ `forgetting` (0.0923–0.0931, overlapping std): **at smoke scale the gate gives no measurable
advantage.** Whether `forgetting_gate` beats `fixed`/`gate` is decided only by the proper-config
run below — if it doesn't, the gate contribution is weak and must be strengthened or re-angled.

## 2. Cold-start / sparsity — `--coldstart 1`

Adds, at the final window, Recall/NDCG/Precision **bucketed by each user's training-interaction
count** (boundaries via `--coldstart_buckets [5,20,50]` → buckets `1-5, 6-20, 21-50, 51+`). Output:
`result/temporal/<tag>_coldstart.json`. Shows how each method does on cold vs. warm users — the
forgetting gate is expected to help most on the **cold** buckets (KG prior for new users/items).

### Smoke demonstration (ml1m, forgetting_gate, T=3, 50 sampled users)

| bucket (train interactions) | users | NDCG@10 | Recall@10 |
|---|---|---|---|
| 1-5 | 4 | 0.2539 | 0.019 |
| 6-20 | 5 | 0.4194 | 0.045 |
| 21-50 | 6 | 0.0510 | 0.010 |
| 51+ | 35 | 0.1469 | 0.064 |

(Very noisy — only a few users land in each bucket at 50 sampled users; use `--topk_users 0`
for stable per-bucket numbers. The mechanism + output (`..._coldstart.json`) are validated.)

## 3. Hyperparameter sensitivity — `scripts/run_temporal_grid.py --experiment sens_*`

| experiment | grid |
|---|---|
| `sens_dim` | dim ∈ {8, 16, 32} |
| `sens_lr` | lr ∈ {0.005, 0.02, 0.05} |
| `sens_window` | n_windows ∈ {5, 8, 10} |
| `sens_replay` | replay_ratio ∈ {0.5, 1.0, 2.0} |
| `sens_ewc` | ewc_lambda ∈ {1, 100, 10000} |

Each writes `result/phase6/<experiment>/summary.{csv,md}` (per-cell mean±std over seeds).

## 4. Interpretability — `--interpret 1`

Exports, for the most active held-out user at the final window, the per-item **fusion gate value
g** (weight on CF vs. KG), the item's **stability features**, and the score →
`result/temporal/<tag>_interpret.json`. This is the case-study data for *why* the gate fuses the
way it does (stable items → higher g / lean CF; cold items → lean KG prior).

### Smoke example (user 1446, forgetting_gate)

| item | gate g (CF weight) | stability [age, freq, recency, low-drift] | score |
|---|---|---|---|
| 630 | 0.216 | [1.0, 1.0, 1.0, 0.72] | 0.915 |
| 2049 | 0.216 | [1.0, 1.0, 1.0, 0.73] | 0.914 |
| 2047 | 0.216 | [1.0, 1.0, 1.0, 0.72] | 0.913 |

(g≈0.22 ⇒ the gate puts ~0.78 weight on the **KG** branch for this user's top items. Full export
in `..._interpret.json`. With proper training the gate should differentiate g across items by
their stability; at smoke scale it is near-uniform.)

## Hand-off: the real Phase 6 (heavy — run on your side, CPU)

```bash
# from repo root; ~runs=3, proper config. Each experiment writes result/phase6/<exp>/summary.md
conda run -n lgkat python scripts/run_temporal_grid.py --experiment ablation \
    --seeds 555,556,557 --dim 32 --windows 10 --epochs 10 --topk_users 0
for E in sens_dim sens_lr sens_window sens_replay sens_ewc; do
  conda run -n lgkat python scripts/run_temporal_grid.py --experiment $E \
      --seeds 555,556,557 --dim 32 --windows 10 --epochs 10 --topk_users 0
done
# cold-start + interpretability (per method, e.g. ours):
cd src && python main.py <proper temporal args> --fusion forgetting_gate --coldstart 1 --interpret 1
```

(`--topk_users 0` = all overlapping users — slower but reportable. EWC λ must be large to have any
effect — see Phase 5.)
