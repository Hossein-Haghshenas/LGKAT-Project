import numpy as np
import scipy.sparse as sp
import os
from time import time


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data, user_item_adj = load_rating(args)
    n_entity, n_relation, adj_entity, adj_relation = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation, user_item_adj

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    '''
    *****************************************
    Rating Matrix for NGCF
    '''
    # keep R sparse end-to-end (dok loop + lil block-assign densifies ~n_user*n_item)
    pos = train_data[train_data[:, 2] > 0.5]
    R = sp.coo_matrix(
        (np.ones(pos.shape[0], dtype=np.float32), (pos[:, 0], pos[:, 1])),
        shape=(n_user, n_item),
        dtype=np.float32,
    ).tocsr()
    R.data[:] = 1.0  # binary like dok overwrite if duplicate (u,i) pairs appear

    plain_adj, norm_adj, mean_adj = get_adj_mat(args, n_user, n_item, R)
    if args.adj_type == 'plain':
        user_item_adj = plain_adj
    elif args.adj_type == 'norm':
        user_item_adj = norm_adj
    elif args.adj_type == 'gcmc':
        user_item_adj = mean_adj
    else:
        user_item_adj = mean_adj + sp.eye(mean_adj.shape[0])

    # paper Table 1: S smoothing steps as A^S on the UI adjacency used by LightGCN
    user_item_adj = adjacency_power(user_item_adj, args.smoothing_steps)
    return n_user, n_item, train_data, eval_data, test_data, user_item_adj


def adjacency_power(adj, power):
    if power is None or power <= 1:
        return adj
    A = adj.tocsr()
    out = A
    for _ in range(int(power) - 1):
        out = out.dot(A)
    return out.tocsr()

'''
*****************************************
From NGCF
'''
def get_adj_mat(args, n_user, n_item, R):
    path = '../data/{}'.format(args.dataset)
    print('Creating UI Graph ...')
    try:
        t1 = time()
        adj_mat = sp.load_npz(path + '/s_adj_mat.npz')
        norm_adj_mat = sp.load_npz(path + '/s_norm_adj_mat.npz')
        mean_adj_mat = sp.load_npz(path + '/s_mean_adj_mat.npz')
        print('already load adj matrix', adj_mat.shape, time() - t1)
    except Exception:
        adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(n_user, n_item, R)
        sp.save_npz(path + '/s_adj_mat.npz', adj_mat)
        sp.save_npz(path + '/s_norm_adj_mat.npz', norm_adj_mat)
        sp.save_npz(path + '/s_mean_adj_mat.npz', mean_adj_mat)
    print('Finish Creating Adjacency Matrix of UI Graph.')
    return adj_mat, norm_adj_mat, mean_adj_mat


def create_adj_mat(n_user, n_item, R):
    # scipy lil block-assign calls toarray() on R → ~11.7GiB densify on restaurant.
    # build the bipartite block via COO index concat (values identical, stays sparse).
    t1 = time()
    R_coo = R.tocoo()
    rows = np.concatenate([R_coo.row, R_coo.col + n_user])
    cols = np.concatenate([R_coo.col + n_user, R_coo.row])
    data = np.concatenate([R_coo.data, R_coo.data]).astype(np.float32)
    adj_mat = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_user + n_item, n_user + n_item),
        dtype=np.float32,
    ).tocsr()
    print('already create adjacency matrix', adj_mat.shape, time() - t1)

    t2 = time()

    def normalized_adj_single(adj):
        '''
        D^(-1)*A
        '''
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    eye = sp.eye(adj_mat.shape[0], dtype=np.float32, format='csr')
    norm_adj_mat = normalized_adj_single(adj_mat + eye)
    mean_adj_mat = normalized_adj_single(adj_mat)

    print('already normalize adjacency matrix', time() - t2)
    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()
'''
*****************************************
'''

def dataset_split(rating_np, args):
    print('---------------')
    print(args.seed)
    print('---------------')
    np.random.seed(args.seed)

    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int64)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)
    adj_entity, adj_relation = construct_adj(args, kg, n_entity)

    return n_entity, n_relation, adj_entity, adj_relation


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = dict()
    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg


def construct_adj(args, kg, entity_num):
    print('constructing adjacency matrix ...')
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    return adj_entity, adj_relation
