# movie WIN: fixed + lr 0.0025 @8192 beats paper 5/5 (proposed_movie_fixed_bs8k).
# (paper batch 65536 OOMs on 16GB; gate+lr=0.02@8192 failed.)

# python main.py --dataset movie --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 8192 --l2_weight 5e-6 --lr 0.0025 --layer_size [32] --adj_type plain --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 1 --pretrain 0 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 8 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag proposed_movie_fixed_bs8k

# --- restaurant (Dianping) ---
# Paper batch 65536 OOMs on GPU. Do NOT copy movie's 8192+0.0025: overfits hard on Dianping.
# Proven-stable: batch 32768 + lr 0.02. If GPU OOM: 16384 + lr 0.01 (not 8192 first).
# create_adj_mat is sparse+cached under data/restaurant/s_*.npz — free ~4GB+ RAM before first build.
# Slow: keep --topk_user_num 100.

python main.py --dataset restaurant --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 32768 --l2_weight 1e-7 --lr 0.02 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --fast_eval 1 --exp_tag proposed_restaurant_fixed_bs32k
