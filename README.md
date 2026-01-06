# A novel recommender system using light graph convolutional network and personalized knowledge-aware attention sub-network

This is our implementation for the following paper:

> Rasoul Hassanzadeh, Vahid Majidneghad, Bahman Arasteh

Corresponding Author: Vahid Majidneghad (vahid.majidnezhad at iau.ac.ir)

## Introduction

Recently, graph neural networks (GNNs) have gained prominence in recommender systems (RS) due to their capability to extract vital features and understand intricate relationships. However, GNNs exhibit limitations in their ability to capture fine-grained semantics in a knowledge graph (KG) and are often insufficient in effectively modeling user-item interactions. One approach to address these limitations is personalized knowledge-aware recommendation. In this paper a novel RS, called LGKAT, is proposed that uses a combination of user-item graph and knowledge graph. It allows for more precise and nuanced modelling of user-item interactions, aiding the recommender system in learning meaningful node representations. One of the contributions of the proposed method is to use a novel integration of light graph convolutional network (LightGCN) in RSs to efficiently manage common signals for user and item embeddings. Another novelty of this paper is to proposed an efficient attention sub-network that encodes rich semantic meanings from the knowledge graph into refined item embeddings in a personalized manner. Extensive tests were conducted on three well-known datasets, demonstrating significant superiority of the proposed method compared to the state-of-the-art methods. The obtained results show that the integration of LightGCN in personalized knowledge-aware recommendation systems can effectively tackle limitations of current recommender systems and improve the quality of recommendations.

## Environment Requirement

The code has been tested running under Python 3.9.12. The required packages are as follows:

- python == 3.9.12
- tensorflow == 2.9.1
- numpy == 1.21.5
- scipy == 1.7.3
- sklearn == 1.0.2

## Examples to Run the code

The instruction of commands has been clearly stated in the code (see src/main.py).

- Movie

```
python main.py  --dataset movie --aggregator concat --n_epochs 10 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 65536 --l2_weight 5e-6 --lr 2e-2 --layer_size [32] --adj_type plain --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 1 --pretrain 0 --att h_ur_t --runs 3 --gpu_id 0
```

- book

```
# Step 1: Preprocess data with PCA (optional but recommended)
python src/preprocess.py -d book --pca_ratings --pca_kg_features --pca_components 50 --pca_variance 0.95

# Step 2: Train the model
python main.py --dataset book --aggregator concat --n_epochs 20 --neighbor_sample_size 8 --dim 64 --n_iter 1 --batch_size 256 --l2_weight 2e-5 --lr 5e-5 --layer_size [64] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 3 --pretrain 0 --att h_ur_t --runs 3 --gpu_id 0
```

Note: If you've already preprocessed the data, you can skip Step 1 and run only Step 2.

- Music

```
python main.py --dataset music --aggregator concat --n_epochs 10 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att h_ur_t --runs 3 --gpu_id 0
```

- Restaurant

```
python main.py --dataset restaurant --aggregator concat --n_epochs 30 --neighbor_sample_size 4 --dim 8 --n_iter 2 --batch_size 65536 --l2_weight 1e-7 --lr 2e-2 --layer_size [8] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --agg_type weighted_avg --smoothing_steps 1 --pretrain 0 --alpha 0.5 --att h_ur_t --runs 3 --gpu_id 0
```

## PCA Preprocessing

The project includes a comprehensive PCA (Principal Component Analysis) preprocessing module for dimensionality reduction. This can be useful for reducing the dimensionality of ratings, knowledge graph features, and embeddings.

### Features

- **Custom PCA Implementation**: A from-scratch PCA implementation for educational purposes
- **Sklearn Integration**: Option to use sklearn's optimized PCA for better performance
- **Multiple Data Types**: Support for ratings, KG features, and embeddings
- **Automatic Component Selection**: Can automatically determine number of components based on variance threshold

### Usage Examples

#### Standalone PCA Preprocessing

Apply PCA to ratings:

```bash
python src/pca_preprocessing.py -t ratings -d movie -n 50
```

Apply PCA to KG features with variance threshold:

```bash
python src/pca_preprocessing.py -t kg_features -d book -v 0.95
```

Apply PCA to embeddings:

```bash
python src/pca_preprocessing.py -t embeddings -i ../output/KGCN_movie_entity_emb.npy -o ../output/KGCN_movie_entity_emb_pca.npy -n 64
```

#### Integrated with Preprocessing Pipeline

Apply PCA during data preprocessing:

```bash
python src/preprocess.py -d movie --pca_ratings --pca_components 50
```

Apply PCA to both ratings and KG features:

```bash
python src/preprocess.py -d book --pca_ratings --pca_kg_features --pca_variance 0.95
```

### Python API Usage

```python
from pca_preprocessing import apply_pca_to_ratings, apply_pca_to_kg_features

# Apply PCA to ratings
reduced_ratings, pca_model = apply_pca_to_ratings(
    rating_file='../data/movie/ratings_final',
    output_file='../data/movie/ratings_final_pca',
    n_components=50,
    variance_threshold=0.95,
    use_sklearn=True,
    standardize=True
)

# Apply PCA to KG features
reduced_features, pca_model = apply_pca_to_kg_features(
    kg_feat_file='../data/book/kg_feat.npz',
    output_file='../data/book/kg_feat_pca',
    n_components=None,  # Auto-determine based on variance
    variance_threshold=0.95,
    use_sklearn=True,
    standardize=True
)
```

For more examples, see `src/example_pca_usage.py`.

## About implementation

We build our model based on the implementations of Personalized knowledge-aware recommendation with collaborative and attentive graph convolutional networks (https://github.com/daiquanyu/COAT).

## About Datasets

The datasets available from the corresponding author

## Citation

If you would like to use our code, please cite:

```

```
