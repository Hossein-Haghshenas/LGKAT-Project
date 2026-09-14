# V2 Book-Crossing residual-fusion SCREEN (locked protocol).
# Frozen: kg_transformer / pool=raw / alpha=0 / no CL / BCE+L2.
# Screening only: --topk_user_num 100 --runs 5.
# Do NOT run all-users unless screening does not falsify residual.
# Run from src/ after: conda activate lgkat
# Sequential: one GPU. Copy one arm at a time if preferred.

# Arm 1 — fixed: e = e_kg
python main.py --dataset book --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 0.0002 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --kgtr_center 0 --kgtr_ffn 0 --kgtr_rel_bias 0 --kgtr_dot 0 --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion fixed --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag v2_book_res_fixed

# Arm 2 — gate: e = g e_cf + (1-g) e_kg
python main.py --dataset book --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 0.0002 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --kgtr_center 0 --kgtr_ffn 0 --kgtr_rel_bias 0 --kgtr_dot 0 --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion gate --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag v2_book_res_gate

# Arm 3 — residual: e = e_kg + g e_cf
python main.py --dataset book --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 0.0002 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --kgtr_center 0 --kgtr_ffn 0 --kgtr_rel_bias 0 --kgtr_dot 0 --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion residual --use_cl 0 --use_simgcl 0 --lr_decay 0.98 --early_stop 5 --early_stop_min_epoch 0 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag v2_book_res_residual
