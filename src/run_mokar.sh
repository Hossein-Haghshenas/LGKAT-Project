#!/usr/bin/env bash
# ===========================================================================
# MOKAR : Multi-Objective Knowledge-Aware Recommendation (NSGA-II over LGKAT)
# Run from the src/ directory:  bash run_mokar.sh
# ===========================================================================
set -e
cd "$(dirname "$0")"

# --- 1) FAST SMOKE TEST (a few minutes; verifies the whole pipeline runs) ---
# Tiny population / generations / epochs just to confirm everything wires up.
python main.py --mode optimize --dataset music --pop_size 6 --n_gen 3 --eval_epochs 3

# --- 2) FULL RUNS (publication-quality; small datasets first) ---------------
# Last.FM (music) and Book-Crossing are small and best suited to the MOO budget.
# python main.py --mode optimize --dataset music --pop_size 20 --n_gen 15 --eval_epochs 8
# python main.py --mode optimize --dataset book  --pop_size 20 --n_gen 15 --eval_epochs 8

# --- 3) SCALABILITY CASE STUDY (larger datasets, reduced budget) ------------
# python main.py --mode optimize --dataset restaurant --pop_size 12 --n_gen 8 --eval_epochs 3
# python main.py --mode optimize --dataset movie      --pop_size 12 --n_gen 8 --eval_epochs 3

# --- 4) REPRODUCIBILITY: repeat with several seeds for significance ---------
# for s in 111 222 333 444 555; do
#   python main.py --mode optimize --dataset music --pop_size 20 --n_gen 15 \
#                  --eval_epochs 8 --moo_seed $s
# done
#
# Outputs are written to ../result/ :
#   mokar_<dataset>_results.json   full archive + Pareto + knee + HV history
#   mokar_<dataset>_pareto.csv     human-readable Pareto front
