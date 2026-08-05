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
        self.n_heads = int(getattr(args, 'n_heads', 4))
        self.kgtr_token = getattr(args, 'kgtr_token', 'rel_ej')
        self.kgtr_pool = getattr(args, 'kgtr_pool', 'transformed')
        if self.att in ('product_mh', 'kg_transformer'):
            if self.dim % self.n_heads != 0:
                raise ValueError(
                    '%s requires dim %% n_heads == 0 (got dim=%d, n_heads=%d)'
                    % (self.att, self.dim, self.n_heads))
        if self.att == 'kg_transformer' and self.kgtr_token not in ('rel_ej', 'product'):
            raise ValueError(
                "kgtr_token must be 'rel_ej' or 'product' (got %r)" % (self.kgtr_token,))
        if self.att == 'kg_transformer' and self.kgtr_pool not in ('transformed', 'raw'):
            raise ValueError(
                "kgtr_pool must be 'transformed' or 'raw' (got %r)" % (self.kgtr_pool,))
        if self.att in ['h_ur_t', 'h_r_t', 'hrt_add', 'ur_ht_mlp', 'u_r_mlp', 'uhrt_concat', 'uhrt_add',
                        'uhrt_add_2', 'uhrt_bi', 'u_h_r_t_mlp', 'product', 'product_mh',
                        'kg_transformer']:
            self.weights_att = self._init_weights_att()
            self.weights_att['att'] = self.att
            self.weights_att['n_heads'] = self.n_heads
            self.weights_att['kgtr_token'] = self.kgtr_token
            self.weights_att['kgtr_pool'] = self.kgtr_pool
        else:
            self.weights_att = dict()
            self.weights_att['att'] = self.att

        # item embeddings aggregation from KGCN and CF
        self.agg_type = args.agg_type
        self.alpha = args.alpha
#        self.beta = args.beta

        # Task 8: cross-view contrastive (InfoNCE) settings. Off by default so the
        # base-model reproduction path is byte-for-byte unchanged.
        self.use_cl = bool(getattr(args, 'use_cl', 0))
        self.cl_lambda = float(getattr(args, 'cl_lambda', 0.1))
        self.cl_tau = float(getattr(args, 'cl_tau', 0.2))
        # SimGCL-style noise CL on CF embeddings (off by default).
        self.use_simgcl = bool(getattr(args, 'use_simgcl', 0))
        self.simgcl_lambda = float(getattr(args, 'simgcl_lambda', 0.1))
        self.simgcl_eps = float(getattr(args, 'simgcl_eps', 0.1))
        if self.agg_type in ['gcn', 'graphsage', 'bi']:
            self.weights_agg = self._init_weights_agg()

        # Task 2(c): adaptive fusion gate vs. fixed-alpha weighted average.
        self.fusion = getattr(args, 'fusion', 'fixed')
        if self.fusion == 'gate' and self.agg_type == 'weighted_avg':
            self.weights_gate = self._init_weights_gate()
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
        # feedable LR (defaults to args.lr). train_n_runs feeds a decayed value when
        # --lr_decay < 1; when unused, Adam sees the same constant as before.
        self.lr_ph = tf.compat.v1.placeholder_with_default(
            tf.constant(self.lr, dtype=tf.float32), shape=[], name='lr_{}'.format(self.model_type))
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
                if self.fusion == 'gate':
                    # Adaptive gate (Task 2c): per-item scalar g = sigmoid(W_g[e_cf;e_kg] + b_g).
                    # Identity init (W_g=0, b_g=logit(alpha)) => g==alpha at step 0, so training
                    # starts exactly at the fixed-alpha weighted average and adapts from there.
                    gate_input = tf.concat([self.item_embeddings_cf, self.item_embeddings_kg], axis=-1)
                    alpha_dynamic = tf.sigmoid(
                        tf.matmul(gate_input, self.weights_gate['gate_W']) + self.weights_gate['gate_b'])
                    self.item_embeddings_final = alpha_dynamic * self.item_embeddings_cf + \
                                                 (1 - alpha_dynamic) * self.item_embeddings_kg
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

        elif self.att == 'product_mh':
            # per-head W on channel-split product features; created only for this att
            head_dim = self.dim // self.n_heads
            shape_w_mh = [3 * head_dim, 1]

        elif self.att == 'kg_transformer':
            # QKV + residual LN over neighbors; created only for this att
            pass

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

            if self.att == 'product_mh':
                for h in range(self.n_heads):
                    all_weights['att_w_mh_%d' % h] = tf.compat.v1.get_variable(
                        shape=shape_w_mh, initializer=KGCN.get_initializer(),
                        name='att_w_mh_%d_%s' % (h, self.model_type))
                    all_weights['att_b_mh_%d' % h] = tf.compat.v1.get_variable(
                        shape=[1], initializer=KGCN.get_initializer(),
                        name='att_b_mh_%d_%s' % (h, self.model_type))

            if self.att == 'kg_transformer':
                d = self.dim
                # rel_ej: concat(r⊙ei, ej); product: concat(u, r⊙ei, ej)
                in_dim = 3 * d if self.kgtr_token == 'product' else 2 * d
                all_weights['att_w_in'] = tf.compat.v1.get_variable(
                    shape=[in_dim, d], initializer=KGCN.get_initializer(),
                    name='att_w_in_{}'.format(self.model_type))
                all_weights['att_b_in'] = tf.compat.v1.get_variable(
                    shape=[d], initializer=KGCN.get_initializer(),
                    name='att_b_in_{}'.format(self.model_type))
                for name in ['att_w_q', 'att_w_k', 'att_w_v', 'att_w_u', 'att_w_o']:
                    all_weights[name] = tf.compat.v1.get_variable(
                        shape=[d, d], initializer=KGCN.get_initializer(),
                        name='{}_{}'.format(name, self.model_type))
                all_weights['att_ln_gamma'] = tf.compat.v1.get_variable(
                    shape=[d], initializer=tf.compat.v1.ones_initializer(),
                    name='att_ln_gamma_{}'.format(self.model_type))
                all_weights['att_ln_beta'] = tf.compat.v1.get_variable(
                    shape=[d], initializer=tf.compat.v1.zeros_initializer(),
                    name='att_ln_beta_{}'.format(self.model_type))

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
    Task 2(c): identity-initialized adaptive fusion gate
    '''

    def _init_weights_gate(self):
        all_weights = dict()

        # Identity initialization: gate_W = 0 and gate_b = logit(alpha) make the gate
        # output exactly alpha at step 0 (equivalent to the fixed-alpha baseline),
        # so the proven base model is the starting point and the gate adapts from there.
        alpha_clamped = min(max(float(self.alpha), 0.01), 0.99)
        bias_init = float(np.log(alpha_clamped / (1.0 - alpha_clamped)))

        with tf.compat.v1.variable_scope('{}'.format(self.model_type)):
            all_weights['gate_W'] = tf.compat.v1.get_variable(
                shape=[2 * self.dim, 1], initializer=tf.compat.v1.zeros_initializer(),
                name='gate_W_{}'.format(self.model_type))
            all_weights['gate_b'] = tf.compat.v1.get_variable(
                initializer=tf.constant([bias_init], dtype=tf.float32),
                name='gate_b_{}'.format(self.model_type))

        return all_weights

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

        self.kg_attention = None  # Task 6: item -> 1-hop KG-neighbor attention [batch, 1, n_neighbor]

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
                # capture the target item's attention over its direct KG neighbors
                if i == 0 and hop == 0:
                    self.kg_attention = aggregator.attention_scores
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

    '''
    *****************************************
    Task 8: cross-view contrastive loss (InfoNCE), paper Section 3-3-6
    '''

    def _create_contrastive_loss(self):
        # The collaborative (CF) and knowledge (KG) views of the SAME item form the
        # positive pair; every other item in the batch is an in-batch negative.
        # Cosine similarity (via L2-normalization) with temperature tau.
        z_cf = tf.nn.l2_normalize(self.item_embeddings_cf, axis=1)  # [batch_size, dim]
        z_kg = tf.nn.l2_normalize(self.item_embeddings_kg, axis=1)  # [batch_size, dim]

        # [batch_size, batch_size]; entry (i, j) = cos(cf_i, kg_j) / tau
        logits = tf.matmul(z_cf, z_kg, transpose_b=True) / self.cl_tau
        labels = tf.range(tf.shape(input=logits)[0])  # positive for row i is column i

        # Symmetric InfoNCE (cf->kg and kg->cf). sparse_softmax_cross_entropy is the
        # numerically stable (log-sum-exp) form of -log( exp(pos) / sum_j exp(.) ).
        loss_cf2kg = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss_kg2cf = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(a=logits))
        return tf.reduce_mean(input_tensor=0.5 * (loss_cf2kg + loss_kg2cf))

    def _noise_augment(self, emb):
        # SimGCL: e' = e + eps * ||e||_2 * (noise / ||noise||_2)
        noise = tf.random.uniform(shape=tf.shape(input=emb), minval=0.0, maxval=1.0, dtype=emb.dtype)
        noise = tf.nn.l2_normalize(noise, axis=1)
        emb_norm = tf.norm(tensor=emb, axis=1, keepdims=True) + 1e-12
        return emb + self.simgcl_eps * emb_norm * noise

    def _infonce_pair(self, z1, z2, tau):
        logits = tf.matmul(z1, z2, transpose_b=True) / tau
        labels = tf.range(tf.shape(input=logits)[0])
        loss_12 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss_21 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=tf.transpose(a=logits))
        return tf.reduce_mean(input_tensor=0.5 * (loss_12 + loss_21))

    def _create_simgcl_loss(self):
        # noise-perturbed CF views (user + item final LightGCN lookups in the batch)
        u1 = tf.nn.l2_normalize(self._noise_augment(self.user_embeddings_final), axis=1)
        u2 = tf.nn.l2_normalize(self._noise_augment(self.user_embeddings_final), axis=1)
        loss_u = self._infonce_pair(u1, u2, self.cl_tau)
        if self.model_type in ['KGCN_NGCF', 'KGCN_LightGCN']:
            i1 = tf.nn.l2_normalize(self._noise_augment(self.item_embeddings_cf), axis=1)
            i2 = tf.nn.l2_normalize(self._noise_augment(self.item_embeddings_cf), axis=1)
        else:
            i1 = tf.nn.l2_normalize(self._noise_augment(self.item_embeddings_final), axis=1)
            i2 = tf.nn.l2_normalize(self._noise_augment(self.item_embeddings_final), axis=1)
        loss_i = self._infonce_pair(i1, i2, self.cl_tau)
        return 0.5 * (loss_u + loss_i)

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
                                           'uhrt_add_2', 'uhrt_bi', 'u_h_r_t_mlp', 'product', 'product_mh',
                                           'kg_transformer']:
                for weights in self.weights_att.keys():
                    if weights.startswith('att_w'):
                        self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_att[weights])

            if self.agg_type in ['gcn', 'graphsage']:
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_agg['agg_w_1'])

            if self.agg_type == 'bi':
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_agg['agg_w_1']) + \
                               tf.nn.l2_loss(self.weights_agg['agg_w_2'])

            if getattr(self, 'fusion', 'fixed') == 'gate' and self.agg_type == 'weighted_avg':
                self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weights_gate['gate_W'])

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

        # Task 8: multi-task objective  L = L_main + cl_lambda * L_cl + l2_weight * L2.
        # cl_loss is always defined (0 when disabled) so the training op can report it.
        self.cl_loss = tf.constant(0.0, dtype=tf.float32)
        if self.use_cl:
            if self.model_type in ['KGCN_NGCF', 'KGCN_LightGCN']:
                self.cl_loss = self._create_contrastive_loss()
                self.loss = self.loss + self.cl_lambda * self.cl_loss
            else:
                print('[CL] --use_cl ignored: model_type {} has no separate CF/KG item views'.format(self.model_type))

        # SimGCL noise CL (independent of cross-view CL; both may be on).
        self.simgcl_loss = tf.constant(0.0, dtype=tf.float32)
        if self.use_simgcl:
            if self.model_type in ['KGCN_NGCF', 'KGCN_LightGCN', 'LightGCN', 'NGCF']:
                self.simgcl_loss = self._create_simgcl_loss()
                self.loss = self.loss + self.simgcl_lambda * self.simgcl_loss
            else:
                print('[SimGCL] --use_simgcl ignored: model_type {} has no CF embeddings'.format(self.model_type))

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr_ph).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.base_loss, self.cl_loss, self.simgcl_loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)

    def supports_cf_cache(self):
        # LightGCN/NGCF tables exist; personalized KG branch (if any) still runs per batch.
        return self.model_type in ['KGCN_LightGCN', 'KGCN_NGCF', 'LightGCN', 'NGCF']

    def cache_cf_embeddings(self, sess):
        # one full-graph CF forward with dropout off; reused for all eval scoring this epoch
        feed_dict = {
            self.node_dropout: [0.] * self.n_layers,
            self.mess_dropout: [0.] * self.n_layers,
        }
        ua, ia = sess.run([self.ua_embeddings, self.ia_embeddings], feed_dict=feed_dict)
        cache = {'ua': ua, 'ia': ia, 'gate_W': None, 'gate_b': None, 'agg': None}
        if self.fusion == 'gate' and self.agg_type == 'weighted_avg':
            cache['gate_W'], cache['gate_b'] = sess.run(
                [self.weights_gate['gate_W'], self.weights_gate['gate_b']])
        if self.agg_type in ['gcn', 'graphsage', 'bi']:
            keys = sorted(self.weights_agg.keys())
            vals = sess.run([self.weights_agg[k] for k in keys])
            cache['agg'] = {k: v for k, v in zip(keys, vals)}
        return cache

    @staticmethod
    def _sigmoid_np(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x, dtype=np.float64)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out.astype(np.float32)

    def _fuse_item_embeddings_np(self, item_cf, item_kg, cache):
        if self.agg_type == 'weighted_avg':
            if self.fusion == 'gate' and cache.get('gate_W') is not None:
                gate_in = np.concatenate([item_cf, item_kg], axis=-1)
                logits = gate_in.dot(cache['gate_W']) + cache['gate_b']
                alpha = self._sigmoid_np(logits)
                return alpha * item_cf + (1.0 - alpha) * item_kg
            return self.alpha * item_cf + (1.0 - self.alpha) * item_kg
        agg = cache.get('agg') or {}
        if self.agg_type == 'gcn':
            h = item_cf + item_kg
            return np.maximum(0.2 * h.dot(agg['agg_w_1']), h.dot(agg['agg_w_1']))
        if self.agg_type == 'graphsage':
            h = np.concatenate([item_cf, item_kg], axis=-1)
            return np.maximum(0.2 * h.dot(agg['agg_w_1']), h.dot(agg['agg_w_1']))
        if self.agg_type == 'bi':
            h1 = item_cf + item_kg
            h2 = item_cf * item_kg
            y1 = np.maximum(0.2 * h1.dot(agg['agg_w_1']), h1.dot(agg['agg_w_1']))
            y2 = np.maximum(0.2 * h2.dot(agg['agg_w_2']), h2.dot(agg['agg_w_2']))
            return y1 + y2
        raise Exception('Unknown agg_type for fast_eval: ' + str(self.agg_type))

    def get_scores_cached(self, sess, feed_dict, cache):
        users = np.asarray(feed_dict[self.user_indices], dtype=np.int64)
        items = np.asarray(feed_dict[self.item_indices], dtype=np.int64)
        user_final = cache['ua'][users]
        item_cf = cache['ia'][items]
        if self.model_type in ['KGCN_LightGCN', 'KGCN_NGCF']:
            item_kg = sess.run(self.item_embeddings_kg, feed_dict)
            item_final = self._fuse_item_embeddings_np(item_cf, item_kg, cache)
        else:
            item_final = item_cf
        logits = np.sum(user_final * item_final, axis=1)
        scores = self._sigmoid_np(logits)
        return items, scores

    def eval_cached(self, sess, feed_dict, cache):
        labels = np.asarray(feed_dict[self.labels])
        _, scores = self.get_scores_cached(sess, feed_dict, cache)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        pred = (scores >= 0.5).astype(np.float32)
        f1 = f1_score(y_true=labels, y_pred=pred)
        return auc, f1

    def get_attention(self, sess, feed_dict):
        # Task 6: item -> direct KG-neighbor attention weights, [batch_size, 1, n_neighbor]
        return sess.run([self.item_indices, self.kg_attention], feed_dict)

    def get_embeddings(self, sess):
        user_emb, entity_emb, relation_emb = sess.run(
            [self.user_emb_matrix, self.entity_emb_matrix, self.relation_emb_matrix])

        return user_emb, entity_emb, relation_emb
