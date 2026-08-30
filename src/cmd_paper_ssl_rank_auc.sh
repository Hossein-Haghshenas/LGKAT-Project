# Optional ranking-stability rerun for book + restaurant only.
# Same architecture as paper_ssl_final_5run; --rank_ckpt auc reports Recall@K
# from the eval-AUC checkpoint (same epoch as F1) and early-stops on eval AUC.
# New exp_tag: does not overwrite paper_ssl_final_5run.
# Run from src/ after: conda activate lgkat

# Book-Crossing
python main.py --dataset book --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 0.0002 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att product --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion gate --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --rank_ckpt auc --fast_eval 1 --log_resources 1 --exp_tag paper_ssl_rank_auc_5run

# Dianping
python main.py --dataset restaurant --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 32768 --l2_weight 1e-7 --lr 0.02 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att product --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --rank_ckpt auc --fast_eval 1 --log_resources 1 --exp_tag paper_ssl_rank_auc_5run

# After both finish, from repo root:
# python scripts/aggregate_final_runs.py --exp_tag paper_ssl_rank_auc_5run --datasets book,restaurant --min_runs 5
