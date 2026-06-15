
# A novel recommender system using light graph convolutional network and personalized knowledge-aware attention sub-network
 
This is our implementation for the following paper:

>Rasoul Hassanzadeh, Vahid Majidneghad, Bahman Arasteh


Corresponding Author: Vahid Majidneghad (vahid.majidnezhad at iau.ac.ir)


## Introduction
Recently, graph neural networks (GNNs) have gained prominence in recommender systems (RS) due to their capability to extract vital features and understand intricate relationships. However, GNNs exhibit limitations in their ability to capture fine-grained semantics in a knowledge graph (KG) and are often insufficient in effectively modeling user-item interactions. One approach to address these limitations is personalized knowledge-aware recommendation. In this paper a novel RS, called LGKAT, is proposed that uses a combination of user-item graph and knowledge graph. It allows for more precise and nuanced modelling of user-item interactions, aiding the recommender system in learning meaningful node representations. One of the contributions of the proposed method is to use a novel integration of light graph convolutional network (LightGCN) in RSs to efficiently manage common signals for user and item embeddings. Another novelty of this paper is to proposed an efficient attention sub-network that encodes rich semantic meanings from the knowledge graph into refined item embeddings in a personalized manner. Extensive tests were conducted on three well-known datasets, demonstrating significant superiority of the proposed method compared to the state-of-the-art methods. The obtained results show that the integration of LightGCN in personalized knowledge-aware recommendation systems can effectively tackle limitations of current recommender systems and improve the quality of recommendations.

## Environment Requirement
The code has been tested running under Python 3.9.12. The required packages are as follows:
* python == 3.9.12
* tensorflow == 2.9.1
* numpy == 1.21.5
* scipy == 1.7.3
* sklearn == 1.0.2

## Examples to Run the code
Place datasets under `data/<dataset_name>/` (see `src/data_loader.py`), then run from `src/`:

```bash
cd src
python main.py ...
```

### Paper reproduction (Product LGKAT, Scientific Reports Tables 2–3)

These commands match the published setup more closely than the older `h_ur_t` examples:

- **`--att product`**: Product attention (paper Eq. 7), the variant reported as **LGKAT (Product)** in the tables.
- **`--n_epochs 39`**: Training length consistent with the paper’s loss curves.
- **`--smoothing_steps`**: Applied as **A^S** on the user–item adjacency before LightGCN (Table 1, coefficient S).
- **`--topk_k_list '[2,10,50,100]'`**: Logs Recall@2 / @10 / @50 / @100 in lockstep with Table 2.
- **`--topk_user_num`**: Default `100` (fast, higher variance vs the paper). Use `0` for **all** users in the train∩test set (slower; on `restaurant` this can be very expensive).
- **`alpha`**: Movie/Book use `alpha=0` below (pure KG item branch + LightGCN users), matching the prior README. The paper’s Eq. (10) uses a blend; for a closer semantic match to the fusion in the article, try **`--alpha 0.5`** on movie/book and compare F1 / Recall.

* Movie
```
python main.py --dataset movie --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 65536 --l2_weight 5e-6 --lr 2e-2 --layer_size [32] --adj_type plain --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 1 --pretrain 0 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100
```

* book
```
python main.py --dataset book --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 5e-5 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100
```

* Music
```
python main.py --dataset music --aggregator concat --n_epochs 39 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100
```

### Improved training (identity-initialized gate + LR decay + early stopping)

The adaptive gate is now identity-initialized (`W_g = 0`, `b_g = logit(alpha)`), so training
starts exactly at the fixed-alpha weighted average and adapts from there. Two new flags:

- **`--lr_decay`** (default `1.0` = off): per-epoch exponential LR decay, e.g. `0.9`.
- **`--early_stop`** (default `0` = off): stop a run if eval AUC has not improved for N epochs
  (reported metrics are always taken from the best eval-AUC epoch, as before).

Recommended commands for the two datasets still below the paper targets:

* Music (target AUC 0.831 / F1 0.742)
```
python main.py --dataset music --aggregator concat --n_epochs 30 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --lr_decay 0.9 --early_stop 8 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att product --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100
```

* Book (target AUC 0.718 / F1 0.650)
```
python main.py --dataset book --aggregator concat --n_epochs 30 --neighbor_sample_size 8 --dim 64 --n_iter 2 --batch_size 256 --l2_weight 5e-5 --lr 2e-4 --lr_decay 0.9 --early_stop 8 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --alpha 0.5 --smoothing_steps 3 --pretrain 0 --att product --runs 5 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100
```

* Restaurant
```
python main.py --dataset restaurant --aggregator concat --n_epochs 39 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 65536 --l2_weight 1e-7 --lr 2e-2 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att product --runs 3 --gpu_id 0 --topk_k_list '[2,10,50,100]' --topk_user_num 100
```

After training, each `result/KGCN_LightGCN_<dataset>_topk_<att>.txt` includes an extra line per epoch: `epoch:N test R@2:... R@10:...` (when those K values are in `--topk_k_list`). The aggregate file `result/KGCN_LightGCN_<dataset>_<att>.txt` ends with a **`paper_style`** mean±std block for R@2 / R@10 / R@50 / R@100 vs Table 2 (Product row) and F1 vs Table 3.

Compare your aggregate log to the paper’s Product LGKAT row (Tables 2–3):

```bash
python scripts/compare_run_to_paper.py result/KGCN_LightGCN_book_product.txt book
```

## About implementation

We build our model based on the implementations of Personalized knowledge-aware recommendation with collaborative and attentive graph convolutional networks (https://github.com/daiquanyu/COAT).

## About Datasets
The datasets available from the corresponding author


## Citation 
If you would like to use our code, please cite:
```

```
