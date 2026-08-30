# Paper-protocol ranking: same SSL recipes as paper_ssl_final_5run, all eligible
# users (--topk_user_num 0). New exp_tag: does not overwrite paper_ssl_final_5run
# or paper_ssl_rank_auc_5run. Default --rank_ckpt r10.
# Run from src/ after: conda activate lgkat
# Copy one dataset at a time. Restaurant all-users is extremely slow.

# Book-Crossing (start here; ranking was the noisy 100-user set)
python main.py --dataset book --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 0.0002 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att product --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 0 --fusion gate --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --fast_eval 1 --log_resources 1 --exp_tag paper_ssl_allusers_5run

# Dianping — all-users top-K is much slower than --topk_user_num 100; leave this running overnight if needed
python main.py --dataset restaurant --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 32768 --l2_weight 1e-7 --lr 0.02 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att product --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 0 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --fast_eval 1 --log_resources 1 --exp_tag paper_ssl_allusers_5run

# Optional: Last.FM (small; same protocol as the paper ranking tables)
python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att product --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 0 --fusion gate --use_cl 1 --cl_lambda 0.02 --cl_tau 0.2 --use_simgcl 1 --simgcl_lambda 0.02 --simgcl_eps 0.05 --lr_decay 0.98 --early_stop 12 --fast_eval 1 --log_resources 1 --exp_tag paper_ssl_allusers_5run

# Optional: MovieLens (all-users eval is heavier than 100 users; batch still 8192)
python main.py --dataset movie --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 8192 --l2_weight 5e-6 --lr 0.0025 --layer_size [32] --adj_type plain --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 1 --pretrain 0 --att product --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 0 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 8 --early_stop_min_epoch 0 --fast_eval 1 --log_resources 1 --exp_tag paper_ssl_allusers_5run

# After the datasets you actually ran, from repo root (drop unused names):
# python scripts/aggregate_final_runs.py --exp_tag paper_ssl_allusers_5run --datasets book,restaurant --min_runs 5
