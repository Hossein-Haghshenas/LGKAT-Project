import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, weights_att, batch_size, dim, n_neighbor, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim
        self.weights_att = weights_att
        self.n_neighbor = n_neighbor
        self.attention_scores = None  # Task 6: last computed [batch, -1, n_neighbor] attention

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        # dimension:
        # self_vectors: [batch_size, -1, dim] ([batch_size, -1] for LabelAggregator)
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim] ([batch_size, -1, n_neighbor] for LabelAggregator)
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        # masks (only for LabelAggregator): [batch_size, -1]
        pass

    def _mix_neighbor_vectors(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, hops):
        if self.weights_att['att'] == 'avg':
            avg = True
        else:
            avg = False

        # avg = True  # no personalized transformation of knowledge graph
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

            # neighbor_vectors: [self.batch_size, -1, self.n_neighbor, self.dim]

            # [batch_size, -1, n_neighbor]
            if self.weights_att['att'] == 'u_r':
                print('Attention: u_r')
                user_relation_scores = tf.reduce_mean(input_tensor=user_embeddings * neighbor_relations, axis=-1)
                user_relation_scores_normalized = tf.compat.v1.nn.softmax(user_relation_scores, axis=-1)

            if self.weights_att['att'] == 'h_r_t':
                print('Attention: h_r_t')
                shape_1 = [-1, self.dim*3]
                shape_2 = [self.batch_size, -1, self.n_neighbor]

                h_e = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                h_e = tf.tile(h_e, [1, 1, self.n_neighbor, 1])

                embeddings = tf.concat([h_e, neighbor_vectors, neighbor_relations], -1)
                embeddings = tf.reshape(embeddings, shape_1)

                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, shape_2)

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'hrt_add':
                print('Attention: hrt_add')
                shape_1 = [-1, self.dim]
                shape_2 = [self.batch_size, -1, self.n_neighbor]

                h_e = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                h_e = tf.tile(h_e, [1, 1, self.n_neighbor, 1])

                embeddings = tf.add_n([h_e, neighbor_vectors, neighbor_relations])
                embeddings = tf.reshape(embeddings, shape_1)

                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, shape_2)

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'u_h_r_t':
                print('Attention: u_h_r_t')

                h_embeddings = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                # h_embeddings = tf.tile(h_embeddings, [1, 1, self.n_neighbor, 1])

                user_relation_scores = tf.reduce_mean(input_tensor=user_embeddings * neighbor_relations, axis=-1)
                target_neigh_scores = tf.reduce_mean(input_tensor=h_embeddings * neighbor_vectors, axis=-1)

                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores+target_neigh_scores, axis=-1)

            if self.weights_att['att'] == 'h_ur_t':
                print('Attention: h_ur_t')
                shape_1 = [-1, self.dim*3]
                shape_2 = [self.batch_size, -1, self.n_neighbor]

                h_embeddings = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                h_embeddings = tf.tile(h_embeddings, [1, 1, self.n_neighbor, 1])

                user_relation_embeddings = user_embeddings * neighbor_relations
                embeddings = tf.concat([h_embeddings, neighbor_vectors, user_relation_embeddings], -1)
                embeddings = tf.reshape(embeddings, shape_1)

                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, shape_2)

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'product':
                # paper Eq.7: concat(u, r⊙ei, ej) then linear score
                print('Attention: product')
                shape_1 = [-1, self.dim * 3]
                shape_2 = [self.batch_size, -1, self.n_neighbor]

                h_embeddings = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                h_embeddings = tf.tile(h_embeddings, [1, 1, self.n_neighbor, 1])

                u_e = tf.tile(user_embeddings, [1, self.n_neighbor ** (hops - 1), self.n_neighbor, 1])
                r_mul_ei = neighbor_relations * h_embeddings
                embeddings = tf.concat([u_e, r_mul_ei, neighbor_vectors], -1)
                embeddings = tf.reshape(embeddings, shape_1)

                scores = tf.nn.leaky_relu(
                    tf.matmul(embeddings, self.weights_att['att_w_1']) + self.weights_att['att_b_1'])
                scores = tf.reshape(scores, shape_2)

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'product_mh':
                # multi-head product scoring (channel-split subspaces); same neighbor mix as product
                print('Attention: product_mh')
                n_heads = int(self.weights_att.get('n_heads', 4))
                head_dim = self.dim // n_heads
                shape_2 = [self.batch_size, -1, self.n_neighbor]

                h_embeddings = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                h_embeddings = tf.tile(h_embeddings, [1, 1, self.n_neighbor, 1])

                u_e = tf.tile(user_embeddings, [1, self.n_neighbor ** (hops - 1), self.n_neighbor, 1])
                # [batch, seq, n_neighbor, n_heads, head_dim]
                u_h = tf.reshape(u_e, [self.batch_size, -1, self.n_neighbor, n_heads, head_dim])
                ei_h = tf.reshape(h_embeddings, [self.batch_size, -1, self.n_neighbor, n_heads, head_dim])
                r_h = tf.reshape(neighbor_relations,
                                 [self.batch_size, -1, self.n_neighbor, n_heads, head_dim])
                ej_h = tf.reshape(neighbor_vectors,
                                  [self.batch_size, -1, self.n_neighbor, n_heads, head_dim])
                z_h = tf.concat([u_h, r_h * ei_h, ej_h], axis=-1)

                scale = tf.sqrt(tf.cast(head_dim, tf.float32))
                head_scores = []
                for h in range(n_heads):
                    z_flat = tf.reshape(z_h[:, :, :, h, :], [-1, 3 * head_dim])
                    s = tf.nn.leaky_relu(
                        tf.matmul(z_flat, self.weights_att['att_w_mh_%d' % h])
                        + self.weights_att['att_b_mh_%d' % h])
                    s = tf.reshape(s, shape_2) / scale
                    head_scores.append(s)

                scores = tf.add_n(head_scores) / float(n_heads)
                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'kg_transformer':
                # personalized QKV self-attn over T neighbors; early-return mix
                kgtr_token = self.weights_att.get('kgtr_token', 'rel_ej')
                kgtr_pool = self.weights_att.get('kgtr_pool', 'transformed')
                kgtr_center = int(self.weights_att.get('kgtr_center', 0))
                kgtr_ffn = int(self.weights_att.get('kgtr_ffn', 0))
                kgtr_temp = float(self.weights_att.get('kgtr_temp', 1.0))
                print('Attention: kg_transformer (token=%s, pool=%s, center=%d, ffn=%d, temp=%.3f)'
                      % (kgtr_token, kgtr_pool, kgtr_center, kgtr_ffn, kgtr_temp))
                n_heads = int(self.weights_att.get('n_heads', 4))
                head_dim = self.dim // n_heads
                t_size = self.n_neighbor

                # [batch, seq, T, dim]
                ei = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])
                ei = tf.tile(ei, [1, 1, t_size, 1])
                r_mul_ei = neighbor_relations * ei
                u_e = tf.tile(user_embeddings, [1, self.n_neighbor ** (hops - 1), t_size, 1])
                if kgtr_token == 'product':
                    tok_in = tf.concat([u_e, r_mul_ei, neighbor_vectors], axis=-1)
                    tok_width = 3 * self.dim
                else:
                    tok_in = tf.concat([r_mul_ei, neighbor_vectors], axis=-1)
                    tok_width = 2 * self.dim
                tok_flat = tf.reshape(tok_in, [-1, tok_width])
                x = tf.matmul(tok_flat, self.weights_att['att_w_in']) + self.weights_att['att_b_in']
                x = tf.reshape(x, [self.batch_size, -1, t_size, self.dim])
                n_seq = tf.shape(x)[1]
                n_flat = self.batch_size * n_seq
                x = tf.reshape(x, [n_flat, t_size, self.dim])

                u = tf.reshape(u_e, [n_flat, t_size, self.dim])
                # center entity (first tile along T is enough; all equal)
                ei_c = tf.reshape(ei, [n_flat, t_size, self.dim])[:, 0:1, :]
                u_c = u[:, 0:1, :]

                k = tf.matmul(x, self.weights_att['att_w_k'])

                def _split_heads(t):
                    # t: [N, S, d] -> [N, H, S, d_h]
                    s = tf.shape(t)[1]
                    t = tf.reshape(t, [n_flat, s, n_heads, head_dim])
                    return tf.transpose(t, [0, 2, 1, 3])

                kh = _split_heads(k)
                scale = tf.sqrt(tf.cast(head_dim, tf.float32)) * kgtr_temp

                if kgtr_center:
                    # Phase-5: single query from center e_i + user
                    q = tf.matmul(ei_c, self.weights_att['att_w_q']) + tf.matmul(
                        u_c, self.weights_att['att_w_u'])
                    qh = _split_heads(q)  # [N, H, 1, d_h]
                    logits = tf.matmul(qh, kh, transpose_b=True) / scale  # [N, H, 1, T]
                    attn = tf.nn.softmax(logits, axis=-1)
                    attn_nb = tf.reduce_mean(attn, axis=1)  # [N, 1, T]
                    attn_nb = tf.squeeze(attn_nb, axis=1)   # [N, T]
                else:
                    q = tf.matmul(x, self.weights_att['att_w_q']) + tf.matmul(
                        u, self.weights_att['att_w_u'])
                    qh = _split_heads(q)
                    logits = tf.matmul(qh, kh, transpose_b=True) / scale
                    attn = tf.nn.softmax(logits, axis=-1)
                    attn_nb = tf.reduce_mean(attn, axis=[1, 2])  # [N, T]

                self.attention_scores = tf.reshape(
                    attn_nb, [self.batch_size, -1, t_size])

                if kgtr_pool == 'raw':
                    nb = tf.reshape(neighbor_vectors, [n_flat, t_size, self.dim])
                    w = tf.expand_dims(attn_nb, axis=-1)
                    neighbors_aggregated = tf.reduce_mean(w * nb, axis=1)
                else:
                    v = tf.matmul(x, self.weights_att['att_w_v'])
                    vh = _split_heads(v)
                    if kgtr_center:
                        oh = tf.matmul(attn, vh)  # [N, H, 1, d_h]
                        oh = tf.transpose(oh, [0, 2, 1, 3])
                        oh = tf.reshape(oh, [n_flat, 1, self.dim])
                        o = tf.matmul(oh, self.weights_att['att_w_o'])
                        y = ei_c + o
                    else:
                        oh = tf.matmul(attn, vh)
                        oh = tf.transpose(oh, [0, 2, 1, 3])
                        oh = tf.reshape(oh, [n_flat, t_size, self.dim])
                        o = tf.matmul(oh, self.weights_att['att_w_o'])
                        y = x + o
                    mean, var = tf.nn.moments(y, axes=[-1], keepdims=True)
                    h = self.weights_att['att_ln_gamma'] * (y - mean) / tf.sqrt(var + 1e-6) \
                        + self.weights_att['att_ln_beta']
                    neighbors_aggregated = tf.reduce_mean(h, axis=1)

                if kgtr_ffn:
                    # residual FFN + LN on pooled neighbor vector
                    h1 = tf.nn.relu(
                        tf.matmul(neighbors_aggregated, self.weights_att['att_w_ffn1'])
                        + self.weights_att['att_b_ffn1'])
                    h2 = tf.matmul(h1, self.weights_att['att_w_ffn2']) + self.weights_att['att_b_ffn2']
                    yf = neighbors_aggregated + h2
                    mean_f, var_f = tf.nn.moments(yf, axes=[-1], keepdims=True)
                    neighbors_aggregated = (
                        self.weights_att['att_ffn_ln_gamma'] * (yf - mean_f)
                        / tf.sqrt(var_f + 1e-6) + self.weights_att['att_ffn_ln_beta'])

                neighbors_aggregated = tf.reshape(
                    neighbors_aggregated, [self.batch_size, -1, self.dim])
                return neighbors_aggregated

            if self.weights_att['att'] == 'ur_ht_mlp':
                # u-r inner product, h-t MLP
                print('Attention: ur_ht_mlp')

                # user-relation
                user_relation_scores = tf.reduce_mean(input_tensor=user_embeddings * neighbor_relations, axis=-1)

                # target-neighbor
                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                h_t = tf.concat([h_e, neighbor_vectors], axis=-1)
                h_t = tf.reshape(h_t, [-1, 2*self.dim])
                
                h_t = tf.nn.leaky_relu(tf.matmul(h_t, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                target_neigh_scores = tf.nn.leaky_relu(tf.matmul(h_t, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                target_neigh_scores = tf.reshape(target_neigh_scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores+target_neigh_scores, axis=-1)

            if self.weights_att['att'] == 'uhrt_concat':
                print('Attention: uhrt_concat')

                u_e = tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1])
                r_e = neighbor_relations

                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                t_e = neighbor_vectors

                embeddings = tf.concat([u_e, r_e, h_e, t_e], axis=-1)
                embeddings = tf.reshape(embeddings, [-1, 4*self.dim])
                
                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'uhrt_add':
                print('Attention: uhrt_add')

                u_e = tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1])
                r_e = neighbor_relations

                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                t_e = neighbor_vectors

                embeddings = tf.add_n([u_e, r_e, h_e, t_e])
                embeddings = tf.reshape(embeddings, [-1, self.dim])
                
                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.reshape(scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'uhrt_add_2':
                # 2-layer MLP
                print('Attention: uhrt_add_2')

                u_e = tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1])
                r_e = neighbor_relations

                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                t_e = neighbor_vectors

                embeddings = tf.add_n([u_e, r_e, h_e, t_e])
                embeddings = tf.reshape(embeddings, [-1, self.dim])
                
                embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                scores = tf.reshape(scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(scores, axis=-1)

            if self.weights_att['att'] == 'uhrt_bi':
                print('Attention: uhrt_bi')
                shape = [-1, self.dim]

                # u_e = tf.reshape(tf.concat([user_embeddings]*(self.n_neighbor**hops), axis=2), shape)
                u_e = tf.reshape(tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1]), shape)
                r_e = tf.reshape(neighbor_relations, shape)

                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                h_e = tf.reshape(h_e, shape)
                t_e = tf.reshape(neighbor_vectors, shape)

                embeddings_1 = tf.add_n([u_e, r_e, h_e, t_e])
                embeddings_2 = u_e * r_e * h_e * t_e

                # embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                # user_relation_scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                embeddings_1 = tf.nn.leaky_relu(tf.matmul(embeddings_1, self.weights_att['att_w_1']))
                embeddings_2 = tf.nn.leaky_relu(tf.matmul(embeddings_2, self.weights_att['att_w_2']))

                user_relation_scores = tf.matmul(embeddings_1 + embeddings_2, self.weights_att['att_w_3'])
                user_relation_scores = tf.reshape(user_relation_scores, [self.batch_size, -1, self.n_neighbor])
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, axis=-1)

            if self.weights_att['att'] == 'u_h_r_t_mlp':
                print('Attention: u_h_r_t_mlp')

                shape = [-1, self.dim]

                # user-relation
                u_e = tf.reshape(tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1]), shape)
                r_e = tf.reshape(neighbor_relations, shape)

                u_r = tf.concat([u_e, r_e], axis=-1)
                u_r = tf.nn.leaky_relu(tf.matmul(u_r, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                user_relation_scores = tf.nn.leaky_relu(tf.matmul(u_r, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                user_relation_scores = tf.reshape(user_relation_scores, [self.batch_size, -1, self.n_neighbor])

                # target-neighbor
                h_e = tf.tile(tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim]), [1, 1, self.n_neighbor, 1])
                h_e = tf.reshape(h_e, shape)
                t_e = tf.reshape(neighbor_vectors, shape)

                h_t = tf.concat([h_e, t_e], axis=-1)
                h_t = tf.nn.leaky_relu(tf.matmul(h_t, self.weights_att['att_w_1_2'])+self.weights_att['att_b_1_2'])
                target_neigh_scores = tf.nn.leaky_relu(tf.matmul(h_t, self.weights_att['att_w_2_2'])+self.weights_att['att_b_2_2'])
                target_neigh_scores = tf.reshape(target_neigh_scores, [self.batch_size, -1, self.n_neighbor])

                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores+target_neigh_scores, axis=-1)

            if self.weights_att['att'] == 'u_r_mlp':
                print('Attention: u_r_mlp')
                shape = [-1, self.dim]

                # u_e = tf.reshape(tf.concat([user_embeddings]*(self.n_neighbor**hops), axis=2), shape)
                u_e = tf.reshape(tf.tile(user_embeddings, [1, self.n_neighbor**(hops-1), self.n_neighbor, 1]), shape)
                r_e = tf.reshape(neighbor_relations, shape)

                embeddings = tf.concat([u_e, r_e], axis=-1)
                embeddings = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_1'])+self.weights_att['att_b_1'])
                user_relation_scores = tf.nn.leaky_relu(tf.matmul(embeddings, self.weights_att['att_w_2'])+self.weights_att['att_b_2'])
                user_relation_scores = tf.reshape(user_relation_scores, [self.batch_size, -1, self.n_neighbor])
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, axis=-1)

            # Task 6: expose the personalized knowledge-aware attention weights over the
            # n_neighbor KG neighbors (per item/entity), shape [batch_size, -1, n_neighbor]
            self.attention_scores = user_relation_scores_normalized

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(input_tensor=user_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(input_tensor=neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, weights, batch_size, dim, n_neighbor, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(weights, batch_size, dim, n_neighbor, dropout, act, name)

        with tf.compat.v1.variable_scope(self.name):
            self.weights = tf.compat.v1.get_variable(
                shape=[self.dim, self.dim], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='weights')
            self.bias = tf.compat.v1.get_variable(shape=[self.dim], initializer=tf.compat.v1.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, hops)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, rate=1 - (1-self.dropout))
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, weights, batch_size, dim, n_neighbor, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(weights, batch_size, dim, n_neighbor, dropout, act, name)

        with tf.compat.v1.variable_scope(self.name):
            self.weights = tf.compat.v1.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='weights')
            self.bias = tf.compat.v1.get_variable(shape=[self.dim], initializer=tf.compat.v1.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, hops)

        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, rate=1 - (1-self.dropout))

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, weights, batch_size, dim, n_neighbor, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(weights, batch_size, dim, n_neighbor, dropout, act, name)

        with tf.compat.v1.variable_scope(self.name):
            self.weights = tf.compat.v1.get_variable(
                shape=[self.dim, self.dim], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name='weights')
            self.bias = tf.compat.v1.get_variable(shape=[self.dim], initializer=tf.compat.v1.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks, hops):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, hops)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, rate=1 - (1-self.dropout))
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)

