# Paper-protocol ranking confirmation (no new architecture).
# Frozen encoder: kg_transformer / pool=raw / paper hop depth.
# Same flags as the 100-user champions except --topk_user_num 0 --runs 5.
# New exp_tag protocol_allusers: does not overwrite 100-user result files.
#
# Run from src/ after: conda activate lgkat
# One GPU. Prefer the sequential runner from repo root:
#   conda run -n lgkat --no-capture-output python scripts/run_protocol_allusers.py
# Default arms: music kgtr, music Product, book kgtr.
# Do NOT start movie/restaurant here unless those Table 2 sentences are kept.
#
# Windows PowerShell (same split/adj RNG as the runner):
#   $env:PYTHONHASHSEED='0'
#   $env:PYTHONUNBUFFERED='1'

# --- required: Last.fm all-users, frozen kgtr/raw (v2_music_hop1 flags) ---
python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --kgtr_center 0 --kgtr_ffn 0 --kgtr_rel_bias 0 --kgtr_dot 0 --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 0 --fusion gate --use_cl 1 --cl_lambda 0.02 --cl_tau 0.2 --use_simgcl 1 --simgcl_lambda 0.02 --simgcl_eps 0.05 --lr_decay 0.98 --early_stop 12 --early_stop_min_epoch 25 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag protocol_allusers

# --- required: Last.fm all-users, Product same stack (paper_ssl_final_5run flags) ---
python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att product --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 0 --fusion gate --use_cl 1 --cl_lambda 0.02 --cl_tau 0.2 --use_simgcl 1 --simgcl_lambda 0.02 --simgcl_eps 0.05 --lr_decay 0.98 --early_stop 12 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag protocol_allusers

# --- next: Book-Crossing all-users, frozen kgtr/raw (gate, no CL) ---
python main.py --dataset book --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 0.0002 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --kgtr_center 0 --kgtr_ffn 0 --kgtr_rel_bias 0 --kgtr_dot 0 --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 0 --fusion gate --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag protocol_allusers

# --- do not run unless a MovieLens Table 2 sentence is kept (n_iter=2 is paper Table 1, not V2) ---
# python main.py --dataset movie --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 8192 --l2_weight 5e-6 --lr 0.0025 --layer_size [32] --adj_type plain --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 1 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --kgtr_center 0 --kgtr_ffn 0 --kgtr_rel_bias 0 --kgtr_dot 0 --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 0 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 8 --early_stop_min_epoch 0 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag protocol_allusers

# --- do not run unless a Dianping Table 2 sentence is kept; all-users is extremely slow ---
# python main.py --dataset restaurant --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 32768 --l2_weight 1e-7 --lr 0.02 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att kg_transformer --n_heads 2 --kgtr_token rel_ej --kgtr_pool raw --kgtr_center 0 --kgtr_ffn 0 --kgtr_rel_bias 0 --kgtr_dot 0 --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 0 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag protocol_allusers

# After music (+ book) finish, from repo root:
# python scripts/aggregate_final_runs.py --exp_tag protocol_allusers --att kg_transformer --datasets music,book --min_runs 5
# python scripts/aggregate_final_runs.py --exp_tag protocol_allusers --att product --datasets music --min_runs 5
# python scripts/compare_run_to_paper.py result/KGCN_LightGCN_music_kg_transformer_protocol_allusers.txt music
# python scripts/compare_run_to_paper.py result/KGCN_LightGCN_music_product_protocol_allusers.txt music
# python scripts/significance_test.py --dataset music --proposed_att kg_transformer --base_att product --proposed_tag protocol_allusers --base_tag protocol_allusers --k 10
# python scripts/compare_run_to_paper.py result/KGCN_LightGCN_book_kg_transformer_protocol_allusers.txt book
