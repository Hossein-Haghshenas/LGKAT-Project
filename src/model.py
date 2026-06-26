import tensorflow as tf
import numpy as np
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score


class KGCN(object):
    def __init__(self, args, n_user, n_item, n_entity, n_relation, adj_entity, adj_relation, user_item_adj, pretrain=None):
        self.pretrain = pretrain
        self._parse_args(args, n_user, n_item, adj_entity, adj_relation, user_item_adj)
        self._build_inputs()
        self._build_model(n_user, n_item, n_entity, n_relation)
        self._build_train()

    tf.compat.v1.disable_eager_execution()

    @staticmethod
    def get_initializer():
        return tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

    def _parse_args(self, args, n_user, n_item, adj_entity, adj_relation, user_item_adj):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
#        self.layer_cl = args.layer_cl
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

        '''
        *****************************************
        CF parameters
        '''
        self.n_user = n_user
        self.n_item = n_item
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type
        self.model_type = args.model_type
        self.smoothing_steps = args.smoothing_steps
        self.norm_adj = user_item_adj  # To do
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag

        # Create Model Parameters (i.e., Initialize Weights).
        if self.model_type in ['KGCN_NGCF', 'KGCN_LightGCN', 'NGCF', 'LightGCN']:
            self.weights_ngcf = self._init_weights_ngcf()

        self.att = args.att  # or 'u_r'
        if self.att in ['h_ur_t', 'h_r_t', 'hrt_add', 'ur_ht_mlp', 'u_r_mlp', 'uhrt_concat', 'uhrt_add',
                        'uhrt_add_2', 'uhrt_bi', 'u_h_r_t_mlp', 'product']:
            self.weights_att = self._init_weights_att()
            self.weights_att['att'] = self.att
        else:
            self.weights_att = dict()
            self.weights_att['att'] = self.att

        # item embeddings aggregation from KGCN and CF
        self.agg_type = args.agg_type
        self.alpha = args.alpha
#        self.beta = args.beta
        if self.agg_type in ['gcn', 'graphsage', 'bi']:
            self.weights_agg = self._init_weights_agg()

        # OL-LGKAT fusion gate (Phase 4). 'fixed' = base alpha blend; 'gate' = adaptive
        # per-item gate; 'forgetting_gate' = gate that also consumes a per-item historical
        # stability signal (unifies CF/KG fusion with the stability-plasticity trade-off).
        self.fusion = getattr(args, 'fusion', 'fixed')
        self.n_stab = int(getattr(args, 'n_stability_feats', 4))
        if self.fusion in ('gate', 'forgetting_gate') and self.agg_type == 'weighted_avg':
            self.weights_gate = self._init_weights_gate()
        self.ewc_lambda = float(getattr(args, 'ewc_lambda', 0.0))   # Phase 5 EWC baseline
        '''
        *****************************************
        '''

    def _build_inputs(self):
        self.user_indices = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None],
                                                     name='user_indices_{}'.format(self.model_type))
        self.item_indices = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None],
                                                     name='item_indices_{}'.format(self.model_type))
        self.labels = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='labels_{}'.format(self.model_type))

        '''
        *****************************************
        NGCF dropout
        '''
        self.node_dropout = tf.compat.v1.placeholder(tf.float32, shape=[None],
                                                     name='node_dropout_{}'.format(self.model_type))
        self.mess_dropout = tf.compat.v1.placeholder(tf.float32, shape=[None],
                                                     name='mess_dropout_{}'.format(self.model_type))
        '''
        *****************************************
        '''

    def _build_model(self, n_user, n_item, n_entity, n_relation):
        with tf.compat.v1.variable_scope('{}'.format(self.model_type)):
            if self.pretrain is not None:
                self.user_emb_matrix = tf.compat.v1.get_variable(initializer=self.pretrain['user'],
                                                                 name='user_emb_matrix_{}'.format(self.model_type))
                self.entity_emb_matrix = tf.compat.v1.get_variable(initializer=self.pretrain['entity'],
                                                                   name='entity_emb_matrix_{}'.format(self.model_type))
                self.relation_emb_matrix = tf.compat.v1.get_variable(initializer=self.pretrain['relation'],
                                                                     name='relation_emb_matrix_{}'.format(
                                                                         self.model_type))
            else:
                self.user_emb_matrix = tf.compat.v1.get_variable(
                    shape=[n_user, self.dim], initializer=KGCN.get_initializer(),
                    name='user_emb_matrix_{}'.format(self.model_type))
                # self.item_emb_matrix = tf.get_variable(
                #     shape=[n_item, self.dim], initializer=KGCN.get_initializer(), name='item_emb_matrix_{}'.format(self.model_type))
                self.entity_emb_matrix = tf.compat.v1.get_variable(
                    shape=[n_entity, self.dim], initializer=KGCN.get_initializer(),
                    name='entity_emb_matrix_{}'.format(self.model_type))
                self.relation_emb_matrix = tf.compat.v1.get_variable(
                    shape=[n_relation, self.dim], initializer=KGCN.get_initializer(),
                    name='relation_emb_matrix_{}'.format(self.model_type))

        # Phase 4: per-item historical stability table (forgetting-aware gate). A NON-trainable
        # variable so it is checkpointed/restored with the model (resume keeps stability intact);
        # written from Python each window via update_stability(), gathered per item in the gate.
        if self.fusion == 'forgetting_gate':
            self.item_stability_table = tf.compat.v1.get_variable(
                shape=[n_item, self.n_stab], initializer=tf.compat.v1.zeros_initializer(),
                trainable=False, name='item_stability_table_{}'.format(self.model_type))
            self.stability_ph = tf.compat.v1.placeholder(
                tf.float32, shape=[n_item, self.n_stab], name='stability_ph_{}'.format(self.model_type))
            self.assign_stability = tf.compat.v1.assign(self.item_stability_table, self.stability_ph)

        '''
        *****************************************
        User Embedding
        '''
        self.user_embeddings = tf.nn.embedding_lookup(params=self.user_emb_matrix, ids=self.user_indices)

        if self.model_type == 'KGCN':
            self.user_embeddings_final = tf.nn.embedding_lookup(params=self.user_emb_matrix, ids=self.user_indices)
            # KGCN for item embeddings
            # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
            # dimensions of entities:
            # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
            entities, relations = self.get_neighbors(self.item_indices)
            # [batch_size, dim]
            self.item_embeddings_final, self.aggregators = self.aggregate(entities, relations)

        elif self.model_type in ['NGCF', 'LightGCN']:
            # ngcf, gcn or gcmc for user and item embeddings
            if self.alg_type == 'ngcf':
                self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
            elif self.alg_type == 'lightgcn':
                self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
            else:
                raise Exception("Unknown alg_type: " + self.alg_type)

            self.user_embeddings_final = tf.nn.embedding_lookup(params=self.ua_embeddings, ids=self.user_indices)
            self.item_embeddings_final = tf.nn.embedding_lookup(params=self.ia_embeddings, ids=self.item_indices)

        elif self.model_type in ['KGCN_NGCF', 'KGCN_LightGCN']:
            # ngcf, gcn or gcmc for user embeddings
            if self.alg_type == 'ngcf':
                self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
            elif self.alg_type == 'lightgcn':
                self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()
            else:
                raise Exception("Unknown alg_type: " + self.alg_type)
            # [batch_size, dim]
            self.user_embeddings_final = tf.nn.embedding_lookup(params=self.ua_embeddings, ids=self.user_indices)
            self.item_embeddings_cf = tf.nn.embedding_lookup(params=self.ia_embeddings, ids=self.item_indices)

            # KGCN for item embeddings
            entities, relations = self.get_neighbors(self.item_indices)
            self.item_embeddings_kg, self.aggregators = self.aggregate(entities, relations)

            # combine item embeddings from CF & KG
            if self.agg_type == 'weighted_avg':
                if self.fusion in ('gate', 'forgetting_gate'):
                    # adaptive per-item gate; 'forgetting_gate' also consumes the stability signal
                    gate_in = tf.concat([self.item_embeddings_cf, self.item_embeddings_kg], axis=-1)
                    if self.fusion == 'forgetting_gate':
                        stab = tf.gather(self.item_stability_table, self.item_indices)
                        gate_in = tf.concat([gate_in, stab], axis=-1)
                    self.fusion_gate = tf.sigmoid(
                        tf.matmul(gate_in, self.weights_gate['gate_W']) + self.weights_gate['gate_b'])
                    self.item_embeddings_final = self.fusion_gate * self.item_embeddings_cf + \
                        (1 - self.fusion_gate) * self.item_embeddings_kg
                else:
                    self.item_embeddings_final = self.alpha * self.item_embeddings_cf + (
                                1 - self.alpha) * self.item_embeddings_kg
            elif self.agg_type == 'gcn':
                # item_embeddings = self.alpha*self.item_embeddings_cf + (1-self.alpha)*self.item_embeddings_kg
                item_embeddings = self.item_embeddings_cf + self.item_embeddings_kg
                self.item_embeddings_final = tf.nn.leaky_relu(tf.matmul(item_embeddings, self.weights_agg['agg_w_1']))
            elif self.agg_type == 'graphsage':
                item_embeddings = tf.concat([self.item_embeddings_cf, self.item_embeddings_kg], axis=-1)
                self.item_embeddings_final = tf.nn.leaky_relu(tf.matmul(item_embeddings, self.weights_agg['agg_w_1']))
            elif self.agg_type == 'bi':
                # item_embeddings_1 = self.alpha*self.item_embeddings_cf + (1-self.alpha)*self.item_embeddings_kg
                item_embeddings_1 = self.item_embeddings_cf + self.item_embeddings_kg
                item_embeddings_2 = tf.multiply(self.item_embeddings_cf, self.item_embeddings_kg)
                self.item_embeddings_final = tf.nn.leaky_relu(
                    tf.matmul(item_embeddings_1, self.weights_agg['agg_w_1'])) + \
                                             tf.nn.leaky_relu(tf.matmul(item_embeddings_2, self.weights_agg['agg_w_2']))

            else:
                raise Exception("Unknown model_type: " + self.model_type)
        '''
        *****************************************
        '''

        # [batch_size]
        self.scores = tf.reduce_sum(input_tensor=self.user_embeddings_final * self.item_embeddings_final, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    '''
    *****************************************
    weights initialization for attention
    '''

    def _init_weights_att(self):
        all_weights = dict()

        if self.att in ['h_ur_t', 'product']:
            shape_w_1 = [3 * self.dim, 1]

        elif self.att == 'h_r_t':
            shape_w_1 = [3 * self.dim, 1]

        elif self.att == 'hrt_add':
            shape_w_1 = [self.dim, 1]

        elif self.att == 'ur_ht_mlp':
            shape_w_1 = [2 * self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

        elif self.att in ['u_r_mlp']:
            shape_w_1 = [2 * self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

        elif self.att in ['u_h_r_t_mlp']:
            shape_w_1 = [2 * self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

            shape_w_1_2 = [2 * self.dim, self.dim]
            shape_w_2_2 = [self.dim, 1]

        elif self.att in ['uhrt_concat']:
            shape_w_1 = [4 * self.dim, 1]

        elif self.att in ['uhrt_add']:
            shape_w_1 = [self.dim, 1]

        elif self.att in ['uhrt_add_2']:
            shape_w_1 = [self.dim, self.dim]
            shape_w_2 = [self.dim, 1]

        elif self.att == 'uhrt_bi':
            shape_w_1 = [self.dim, self.dim]
            shape_w_2 = [self.dim, self.dim]
            shape_w_3 = [self.dim, 1]

        with tf.compat.v1.variable_scope('{}'.format(self.model_type)):

            if self.att in ['h_r_t', 'h_ur_t', 'product', 'uhrt_concat', 'uhrt_add']:
                all_weights['att_w_1'] = tf.compat.v1.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

            if self.att in ['uhrt_bi']:
                all_weights['att_w_1'] = tf.compat.v1.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.compat.v1.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

                all_weights['att_w_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_2, initializer=KGCN.get_initializer(), name='att_w_2_{}'.format(self.model_type))
                all_weights['att_b_2'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_{}'.format(self.model_type))

                all_weights['att_w_3'] = tf.compat.v1.get_variable(
                    shape=shape_w_3, initializer=KGCN.get_initializer(), name='att_w_3_{}'.format(self.model_type))
                all_weights['att_b_3'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_3_{}'.format(self.model_type))

            if self.att in ['ur_ht_mlp', 'u_r_mlp', 'uhrt_add_2']:
                all_weights['att_w_1'] = tf.compat.v1.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.compat.v1.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

                all_weights['att_w_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_2, initializer=KGCN.get_initializer(), name='att_w_2_{}'.format(self.model_type))
                all_weights['att_b_2'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_{}'.format(self.model_type))

            if self.att in ['u_h_r_t_mlp']:
                all_weights['att_w_1'] = tf.compat.v1.get_variable(
                    shape=shape_w_1, initializer=KGCN.get_initializer(), name='att_w_1_{}'.format(self.model_type))
                all_weights['att_b_1'] = tf.compat.v1.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_{}'.format(self.model_type))

                all_weights['att_w_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_2, initializer=KGCN.get_initializer(), name='att_w_2_{}'.format(self.model_type))
                all_weights['att_b_2'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_{}'.format(self.model_type))

                all_weights['att_w_1_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_1_2, initializer=KGCN.get_initializer(), name='att_w_1_2_{}'.format(self.model_type))
                all_weights['att_b_1_2'] = tf.compat.v1.get_variable(
                    shape=[1, self.dim], initializer=KGCN.get_initializer(), name='att_b_1_2{}'.format(self.model_type))

                all_weights['att_w_2_2'] = tf.compat.v1.get_variable(
                    shape=shape_w_2_2, initializer=KGCN.get_initializer(), name='att_w_2_2_{}'.format(self.model_type))
                all_weights['att_b_2_2'] = tf.compat.v1.get_variable(
                    shape=[1], initializer=KGCN.get_initializer(), name='att_b_2_2_{}'.format(self.model_type))

        return all_weights

    '''
    *****************************************
    '''

    '''
    *****************************************
    weights initialization for combining item embeddings
    '''

    def _init_weights_agg(self):
        all_weights = dict()

        if self.agg_type in ['gcn', 'bi']:
            shape_w = [self.dim, self.dim]
            shape_b = [1, self.dim]
        elif self.agg_type == 'graphsage':
            shape_w = [2 * self.dim, self.dim]
            shape_b = [1, self.dim]
        else:
            raise Exception('Unknown  agg_type: {}'.format(self.agg_type))

        with tf.compat.v1.variable_scope('{}'.format(self.model_type)):
            all_weights['agg_w_1'] = tf.compat.v1.get_variable(
                shape=shape_w, initializer=KGCN.get_initializer(), name='agg_w_1_{}'.format(self.model_type))
            all_weights['agg_b_1'] = tf.compat.v1.get_variable(
                shape=shape_b, initializer=KGCN.get_initializer(), name='agg_b_1_{}'.format(self.model_type))

            if self.agg_type == 'bi':
                all_weights['agg_w_2'] = tf.compat.v1.get_variable(
                    shape=shape_w, initializer=KGCN.get_initializer(), name='agg_w_2_{}'.format(self.model_type))
                all_weights['agg_b_2'] = tf.compat.v1.get_variable(
                    shape=shape_b, initializer=KGCN.get_initializer(), name='agg_b_2_{}'.format(self.model_type))

        return all_weights

    '''
    *****************************************
    Phase 4: forgetting-aware fusion gate (identity-initialized)
    '''
    def _init_weights_gate(self):
        # Identity init: gate_W = 0, gate_b = logit(alpha) => the gate outputs exactly alpha at
        # step 0 (== fixed-alpha baseline), so 'gate'/'forgetting_gate' start at the proven base
        # model and adapt from there. 'forgetting_gate' widens the input with n_stab features.
        in_dim = 2 * self.dim + (self.n_stab if self.fusion == 'forgetting_gate' else 0)
        a = min(max(float(self.alpha), 0.01), 0.99)
        bias_init = float(np.log(a / (1.0 - a)))
        all_weights = dict()
        with tf.compat.v1.variable_scope('{}'.format(self.model_type)):
            all_weights['gate_W'] = tf.compat.v1.get_variable(
                shape=[in_dim, 1], initializer=tf.compat.v1.zeros_initializer(),
                name='gate_W_{}'.format(self.model_type))
            all_weights['gate_b'] = tf.compat.v1.get_variable(
                initializer=tf.constant([bias_init], dtype=tf.float32),
                name='gate_b_{}'.format(self.model_type))
        return all_weights

    '''
    *****************************************
    '''

    '''
    *****************************************
    user and item embedding with NGCF
    '''

    def _init_weights_ngcf(self):
        all_weights = dict()

        self.weight_size_list = [self.dim] + self.weight_size

        for k in range(self.n_layers):

            with tf.compat.v1.variable_scope('{}'.format(self.model_type)):

                all_weights['W_gc_%d' % k] = tf.compat.v1.get_variable(
                    shape=[self.weight_size_list[k], self.weight_size_list[k + 1]], initializer=KGCN.get_initializer(),
                    name='W_gc_{}_{}'.format(k, self.model_type))
                all_weights['b_gc_%d' % k] = tf.compat.v1.get_variable(
                    shape=[1, self.weight_size_list[k + 1]], initializer=KGCN.get_initializer(),
                    name='b_gc_{}_{}'.format(k, self.model_type))

                if self.alg_type == 'ngcf':
                    all_weights['W_bi_%d' % k] = tf.compat.v1.get_variable(
                        shape=[self.weight_size_list[k], self.weight_size_list[k + 1]],
                        initializer=KGCN.get_initializer(), name='W_bi_{}_{}'.format(k, self.model_type))
                    all_weights['b_bi_%d' % k] = tf.compat.v1.get_variable(
                        shape=[1, self.weight_size_list[k + 1]], initializer=KGCN.get_initializer(),
                        name='b_bi_{}_{}'.format(k, self.model_type))

                if self.alg_type == 'lightgcn':
                    all_weights['W_lt_%d' % k] = tf.compat.v1.get_variable(
                        shape=[self.weight_size_list[k], self.weight_size_list[k + 1]],
                        initializer=KGCN.get_initializer(), name='W_lt_{}_{}'.format(k, self.model_type))
                    all_weights['b_lt_%d' % k] = tf.compat.v1.get_variable(
                        shape=[1, self.weight_size_list[k + 1]], initializer=KGCN.get_initializer(),
                        name='b_lt_{}_{}'.format(k, self.model_type))

        return all_weights

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(X, dropout_mask)

        return pre_out * tf.compat.v1.div(1., keep_prob)

    def _split_A_hat(self, X):
        A_fold_hat = []
        n_fold = 100
        fold_len = (self.n_user + self.n_item) // n_fold
        for i_fold in range(n_fold):
            start = i_fold * fold_len
            if i_fold == n_fold - 1:
                end = self.n_user + self.n_item
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat


    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []
        n_fold = 100
        fold_len = (self.n_user + self.n_item) // n_fold
        for i_fold in range(n_fold):
            start = i_fold * fold_len
            if i_fold == n_fold - 1:
                end = self.n_user + self.n_item
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed(self):
        n_fold = 100
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.user_emb_matrix, self.entity_emb_matrix[:self.n_item, :]], axis=0)  # E
        #ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(n_fold):
                temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_user, self.n_item], 0)
        return u_g_embeddings, i_g_embeddings


    def _create_ngcf_embed(self):
        if self.node_dropout_flag:
            # node dropout.
            temp = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
            A = self._dropout_sparse(temp, 1 - self.node_dropout[0], self.n_nonzero_elems)
        else:
            A = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.user_emb_matrix, self.entity_emb_matrix[:self.n_item, :]], axis=0)  # E
        # ego_embeddings = tf.concat([self.user_emb_matrix, self.item_emb_matrix], axis=0)  # E

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            side_embeddings = tf.sparse.sparse_dense_matmul(A, ego_embeddings)  # Eq. (7) L*E
            # transformed sum messages of neighbors.
            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(side_embeddings, self.weights_ngcf['W_gc_%d' % k]) + self.weights_ngcf[
                    'b_gc_%d' % k])  # Eq. (7) L*E*W

            # bi messages of neighbors.
            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)  # Eq. (7) LE element_wise_dot E
            # transformed bi messages of neighbors.
            # Eq. (7) LE element_wise_dot E*W
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights_ngcf['W_bi_%d' % k]) + self.weights_ngcf['b_bi_%d' % k])

            # non-linear activation.
            ego_embeddings = sum_embeddings + bi_embeddings

            # message dropout.
            ego_embeddings = tf.nn.dropout(ego_embeddings, rate=1 - (1 - self.mess_dropout[k]))

            # normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]
        # Method 1
        all_embeddings = tf.concat(all_embeddings, 1)  # Eq. (9)
        # # Method 2
        # all_embeddings = all_embeddings[-1]
        # # Method 3
        # all_embeddings = tf.add_n(all_embeddings)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_user, self.n_item], 0)
        return u_g_embeddings, i_g_embeddings


        '''
    *****************************************
    '''

    '''
    ******************************************
    item embedding in knowledge graph with KGCN
    '''

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(params=self.entity_emb_matrix, ids=i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(params=self.relation_emb_matrix, ids=i) for i in relations]

        res = [tf.reshape(entity_vectors[0], [self.batch_size, self.dim])]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.weights_att, self.batch_size, self.dim,
                                                   self.n_neighbor, act=tf.nn.tanh,
                                                   name=self.model_type + '_{}'.format(i))
            else:
                aggregator = self.aggregator_class(self.weights_att, self.batch_size, self.dim,
                                                   self.n_neighbor, name=self.model_type + '_{}'.format(i))
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings,
                                    masks=None,
                                    hops=hop + 1)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
            res.append(tf.reshape(entity_vectors[0], [self.batch_size, self.dim]))

        # res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        # # Method 1
        # res = tf.concat(res, 1)

        # Method 2
        res = tf.add_n(res)

        # # Method 3
        # res = res[-1]

        return res, aggregators

    '''
    *****************************************
    '''

    def _build_train(self):
        self.base_loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        # self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(self.entity_emb_matrix) + \
        #         tf.nn.l2_loss(self.relation_emb_matrix) + tf.nn.l2_loss(self.item_emb_matrix)

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(self.entity_emb_matrix) + \
                       tf.nn.l2_loss(self.relation_emb_matrix)

        if self.model_type in ['KGCN', 'KGCN_NGCF', 'KGCN_LightGCN']:
            for aggregator in self.aggregators:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

            if self.weights_att['att'] in ['h_ur_t', 'h_r_t', 'ur_ht_mlp', 'u_r_mlp', 'uhrt_concat', 'uhrt_add',
                                           'uhrt_add_2', 'uhrt_bi', 'u_h_r_t_mlp', 'product']:
                for weights in self.weights_att.keys():
                    if weights.startswith('att_w'):
                        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_att[weights])

            if getattr(self, 'fusion', 'fixed') in ('gate', 'forgetting_gate') and self.agg_type == 'weighted_avg':
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_gate['gate_W'])

            if self.agg_type in ['gcn', 'graphsage']:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_agg['agg_w_1'])

            if self.agg_type == 'bi':
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_agg['agg_w_1']) + \
                               tf.nn.l2_loss(self.weights_agg['agg_w_2'])

        '''
        ******************************************
        L2 loss in NGCF
        '''
        # l2_loss_ngcf = []
        # for key in self.weights_ngcf:
        #     if 'W' in key:
        #         l2_loss_ngcf.append(tf.nn.l2_loss(self.weights_ngcf[key]))
        # l2_loss_ngcf = tf.add_n(l2_loss_ngcf)

        # self.l2_loss = self.l2_loss + l2_loss_ngcf
        '''
        *****************************************
        '''

        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        # Phase 5 EWC baseline: elastic weight consolidation penalty on the embeddings
        # (lambda/2 * sum F * (theta - theta_star)^2). Off (graph unchanged) unless ewc_lambda > 0.
        if getattr(self, 'ewc_lambda', 0.0) > 0:
            self._build_ewc()
            self.loss = self.loss + self.ewc_penalty

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.base_loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)

    def get_embeddings(self, sess):
        user_emb, entity_emb, relation_emb = sess.run(
            [self.user_emb_matrix, self.entity_emb_matrix, self.relation_emb_matrix])

        return user_emb, entity_emb, relation_emb

    def update_stability(self, sess, stability_matrix):
        """Phase 4: write the per-item stability table (shape [n_item, n_stab]) from Python.
        No-op unless fusion == 'forgetting_gate'."""
        if getattr(self, 'assign_stability', None) is not None:
            sess.run(self.assign_stability, feed_dict={self.stability_ph: stability_matrix})

    '''
    *****************************************
    Phase 5: EWC (elastic weight consolidation) on the embedding matrices
    '''
    def _build_ewc(self):
        self._ewc_params = {'user': self.user_emb_matrix, 'entity': self.entity_emb_matrix,
                            'relation': self.relation_emb_matrix}
        self.ewc_F = {}; self.ewc_star = {}
        self.ewc_F_ph = {}; self.ewc_star_ph = {}; self.ewc_assign = []
        penalty = 0.0
        with tf.compat.v1.variable_scope('{}_ewc'.format(self.model_type)):
            for name, p in self._ewc_params.items():
                F = tf.compat.v1.get_variable('F_' + name, shape=p.shape,
                    initializer=tf.compat.v1.zeros_initializer(), trainable=False)
                star = tf.compat.v1.get_variable('star_' + name, shape=p.shape,
                    initializer=tf.compat.v1.zeros_initializer(), trainable=False)
                F_ph = tf.compat.v1.placeholder(tf.float32, p.shape)
                star_ph = tf.compat.v1.placeholder(tf.float32, p.shape)
                self.ewc_F[name] = F; self.ewc_star[name] = star
                self.ewc_F_ph[name] = F_ph; self.ewc_star_ph[name] = star_ph
                self.ewc_assign += [tf.compat.v1.assign(F, F_ph), tf.compat.v1.assign(star, star_ph)]
                penalty = penalty + tf.reduce_sum(input_tensor=F * tf.square(p - star))
        self.ewc_penalty = 0.5 * self.ewc_lambda * penalty
        # Fisher = (d base_loss / d theta)^2. A param the loss does not depend on (e.g. relation
        # embeddings under model_type LightGCN) has grad None -> use zeros so EWC stays robust.
        names = ['user', 'entity', 'relation']
        g = tf.gradients(ys=self.base_loss, xs=[self._ewc_params[n] for n in names])
        self.ewc_fisher = {n: (tf.square(gi) if gi is not None else tf.zeros_like(self._ewc_params[n]))
                           for n, gi in zip(names, g)}

    def compute_fisher(self, sess, feed_dict):
        return sess.run(self.ewc_fisher, feed_dict)

    def get_param_values(self, sess):
        u, e, r = sess.run([self.user_emb_matrix, self.entity_emb_matrix, self.relation_emb_matrix])
        return {'user': u, 'entity': e, 'relation': r}

    def set_ewc(self, sess, F_dict, star_dict):
        fd = {}
        for name in self.ewc_F_ph:
            fd[self.ewc_F_ph[name]] = F_dict[name]
            fd[self.ewc_star_ph[name]] = star_dict[name]
        sess.run(self.ewc_assign, feed_dict=fd)
