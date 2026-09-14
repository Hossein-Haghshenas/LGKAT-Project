# V2 Last.fm 2-hop SCREEN (locked topology hypothesis).
# Frozen: kg_transformer / pool=raw / fusion=gate / CL+SimGCL / BCE+L2.
# Only n_iter changes. Screening: --topk_user_num 100 --runs 5.
# Run from src/ after: conda activate lgkat

# Arm 1 — n_iter=1
python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --kgtr_center 0 --kgtr_ffn 0 --kgtr_rel_bias 0 --kgtr_dot 0 --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion gate --use_cl 1 --cl_lambda 0.02 --cl_tau 0.2 --use_simgcl 1 --simgcl_lambda 0.02 --simgcl_eps 0.05 --lr_decay 0.98 --early_stop 12 --early_stop_min_epoch 25 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag v2_music_hop1

# Arm 2 — n_iter=2
python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 2 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att kg_transformer --n_heads 4 --kgtr_token rel_ej --kgtr_pool raw --kgtr_center 0 --kgtr_ffn 0 --kgtr_rel_bias 0 --kgtr_dot 0 --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100 --fusion gate --use_cl 1 --cl_lambda 0.02 --cl_tau 0.2 --use_simgcl 1 --simgcl_lambda 0.02 --simgcl_eps 0.05 --lr_decay 0.98 --early_stop 12 --early_stop_min_epoch 25 --rank_ckpt r10 --fast_eval 1 --log_resources 1 --exp_tag v2_music_hop2
