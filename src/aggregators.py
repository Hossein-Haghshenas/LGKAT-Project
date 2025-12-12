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
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None, hops=None):
        outputs = self._call(self_vectors, neighbor_vectors,
                             neighbor_relations, user_embeddings, masks, hops)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None, hops=None):
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings, hops=None):
        avg = True
        if not avg:
            user_embeddings = tf.reshape(
                user_embeddings, [self.batch_size, 1, 1, self.dim])
            user_relation_scores = tf.reduce_mean(input_tensor=user_embeddings * neighbor_relations, axis=-1,
                                                  keepdims=True)
            user_relation_scores_normalized = tf.nn.softmax(
                user_relation_scores, axis=2)
            neighbors_aggregated = tf.reduce_mean(input_tensor=user_relation_scores_normalized * neighbor_vectors,
                                                  axis=2)
        else:
            neighbors_aggregated = tf.reduce_mean(
                input_tensor=neighbor_vectors, axis=2)
        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, weights_att, batch_size, dim, n_neighbor, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(
            batch_size, dim, dropout, act, name)

        self.n_neighbor = n_neighbor

        with tf.compat.v1.variable_scope(self.name):
            self.weights = tf.compat.v1.get_variable(
                shape=[self.dim, self.dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                            distribution="uniform"),
                name='weights')
            self.bias = tf.compat.v1.get_variable(
                shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None, hops=None):
        neighbors_agg = self._mix_neighbor_vectors(
            neighbor_vectors, neighbor_relations, user_embeddings, hops)
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, rate=1 - (1 - self.dropout))
        output = tf.matmul(output, self.weights) + self.bias
        output = tf.reshape(output, [self.batch_size, -1, self.dim])
        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, weights_att, batch_size, dim, n_neighbor, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(
            batch_size, dim, dropout, act, name)

        self.n_neighbor = n_neighbor

        with tf.compat.v1.variable_scope(self.name):
            self.weights = tf.compat.v1.get_variable(
                shape=[self.dim * 2, self.dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                            distribution="uniform"),
                name='weights')
            self.bias = tf.compat.v1.get_variable(
                shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None, hops=None):
        neighbors_agg = self._mix_neighbor_vectors(
            neighbor_vectors, neighbor_relations, user_embeddings, hops)
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, rate=1 - (1 - self.dropout))
        output = tf.matmul(output, self.weights) + self.bias
        output = tf.reshape(output, [self.batch_size, -1, self.dim])
        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, weights_att, batch_size, dim, n_neighbor, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(
            batch_size, dim, dropout, act, name)

        self.n_neighbor = n_neighbor

        with tf.compat.v1.variable_scope(self.name):
            self.weights = tf.compat.v1.get_variable(
                shape=[self.dim, self.dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                            distribution="uniform"),
                name='weights')
            self.bias = tf.compat.v1.get_variable(
                shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None, hops=None):
        neighbors_agg = self._mix_neighbor_vectors(
            neighbor_vectors, neighbor_relations, user_embeddings, hops)
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, rate=1 - (1 - self.dropout))
        output = tf.matmul(output, self.weights) + self.bias
        output = tf.reshape(output, [self.batch_size, -1, self.dim])
        return self.act(output)


class MultiHeadAttentionAggregator(Aggregator):
    """
    Multi-head attention aggregator for knowledge-aware recommendation
    """

    def __init__(self, weights_att, batch_size, dim, n_neighbor, n_heads=4, dropout=0., act=tf.nn.relu, name=None):
        super(MultiHeadAttentionAggregator, self).__init__(
            batch_size, dim, dropout, act, name)

        self.n_neighbor = n_neighbor
        self.n_heads = n_heads
        self.weights_att = weights_att
        self.att_type = weights_att.get('att', 'u_r')

        # Ensure dimension is divisible by number of heads
        assert dim % n_heads == 0, f"Dimension {dim} must be divisible by number of heads {n_heads}"
        self.head_dim = dim // n_heads

        with tf.compat.v1.variable_scope(self.name):
            # Multi-head projection matrices
            self.W_query = tf.compat.v1.get_variable(
                shape=[self.dim, self.dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                            distribution="uniform"),
                name='W_query')
            self.W_key = tf.compat.v1.get_variable(
                shape=[self.dim, self.dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                            distribution="uniform"),
                name='W_key')
            self.W_value = tf.compat.v1.get_variable(
                shape=[self.dim, self.dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                            distribution="uniform"),
                name='W_value')

            # Relation-aware projection
            self.W_relation = tf.compat.v1.get_variable(
                shape=[self.dim, self.dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                            distribution="uniform"),
                name='W_relation')

            # Output projection
            self.W_output = tf.compat.v1.get_variable(
                shape=[self.dim, self.dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                            distribution="uniform"),
                name='W_output')
            self.bias_output = tf.compat.v1.get_variable(
                shape=[self.dim],
                initializer=tf.zeros_initializer(),
                name='bias_output')

            # Layer normalization parameters
            self.gamma = tf.compat.v1.get_variable(
                shape=[self.dim],
                initializer=tf.ones_initializer(),
                name='gamma')
            self.beta = tf.compat.v1.get_variable(
                shape=[self.dim],
                initializer=tf.zeros_initializer(),
                name='beta')

        # Store weights for L2 regularization
        self.weights = self.W_query

    def _split_heads(self, x):
        """Split the last dimension into (n_heads, head_dim)"""
        # x: [batch, length, dim]
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        x = tf.reshape(x, [batch_size, length, self.n_heads, self.head_dim])
        # [batch, n_heads, length, head_dim]
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x):
        """Merge the head dimension"""
        # x: [batch, n_heads, length, head_dim]
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])  # [batch, length, n_heads, head_dim]
        return tf.reshape(x, [batch_size, length, self.dim])

    def _compute_attention(self, queries, keys, values, relations, user_embeddings):
        """
        Compute multi-head attention with relation and user awareness
        """
        # Standard scaled dot-product attention
        # [batch, n_heads, seq_len, n_neighbor]
        matmul_qk = tf.matmul(queries, keys, transpose_b=True)

        # Scale
        dk = tf.cast(self.head_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)

        # Add relation bias - FIXED
        # [batch, n_heads, seq_len, 1]
        relation_bias = tf.reduce_sum(
            queries * relations, axis=-1, keepdims=True)
        # [batch, n_heads, seq_len, n_neighbor]
        relation_bias = tf.tile(relation_bias, [1, 1, 1, tf.shape(keys)[2]])

        # Add user preference bias - FIXED
        user_emb_heads = tf.reshape(
            user_embeddings, [self.batch_size, 1, 1, self.dim])
        user_emb_heads = tf.tile(
            user_emb_heads, [1, 1, tf.shape(queries)[2], 1])
        # [batch, n_heads, seq_len, head_dim]
        user_emb_split = self._split_heads(user_emb_heads)

        # Compute user bias by comparing with keys
        # [batch, n_heads, seq_len, n_neighbor]
        user_bias = tf.matmul(user_emb_split, keys, transpose_b=True)
        user_bias = user_bias / tf.sqrt(dk)  # Scale it too

        # Combine all attention signals
        attention_logits = scaled_attention_logits + \
            0.5 * relation_bias + 0.3 * user_bias

        # Softmax
        # [batch, n_heads, seq_len, n_neighbor]
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)

        # Apply dropout to attention weights
        attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout)

        # Apply attention to values
        # [batch, n_heads, seq_len, head_dim]
        output = tf.matmul(attention_weights, values)

        return output, attention_weights

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None, hops=None):
        """
        self_vectors: [batch_size, seq_len, dim]
        neighbor_vectors: [batch_size, seq_len, n_neighbor, dim]
        neighbor_relations: [batch_size, seq_len, n_neighbor, dim]
        user_embeddings: [batch_size, dim]
        """
        # Get dynamic shapes
        batch_size = tf.shape(self_vectors)[0]
        seq_len = tf.shape(self_vectors)[1]
        n_neighbor = tf.shape(neighbor_vectors)[2]

        # Flatten neighbor dimensions for processing
        # neighbor_vectors: [batch_size, seq_len, n_neighbor, dim]
        # We want to keep seq_len separate and only process neighbors
        # Reshape to [batch_size * seq_len, n_neighbor, dim]
        neighbor_vectors_reshaped = tf.reshape(
            neighbor_vectors, [batch_size * seq_len, n_neighbor, self.dim])
        neighbor_relations_reshaped = tf.reshape(
            neighbor_relations, [batch_size * seq_len, n_neighbor, self.dim])

        # Reshape self_vectors to [batch_size * seq_len, 1, dim] for broadcasting
        self_vectors_reshaped = tf.reshape(
            self_vectors, [batch_size * seq_len, 1, self.dim])

        # Linear projections
        # Queries from self_vectors: [batch_size * seq_len, 1, dim]
        queries = tf.reshape(self_vectors_reshaped, [-1, self.dim])
        queries = tf.matmul(queries, self.W_query)
        queries = tf.reshape(queries, [batch_size * seq_len, 1, self.dim])

        # Keys from neighbors: [batch_size * seq_len, n_neighbor, dim]
        keys = tf.reshape(neighbor_vectors_reshaped, [-1, self.dim])
        keys = tf.matmul(keys, self.W_key)
        keys = tf.reshape(keys, [batch_size * seq_len, n_neighbor, self.dim])

        # Values from neighbors: [batch_size * seq_len, n_neighbor, dim]
        values = tf.reshape(neighbor_vectors_reshaped, [-1, self.dim])
        values = tf.matmul(values, self.W_value)
        values = tf.reshape(
            values, [batch_size * seq_len, n_neighbor, self.dim])

        # Relations: [batch_size * seq_len, n_neighbor, dim]
        relations = tf.reshape(neighbor_relations_reshaped, [-1, self.dim])
        relations = tf.matmul(relations, self.W_relation)
        relations = tf.reshape(
            relations, [batch_size * seq_len, n_neighbor, self.dim])

        # Split into multiple heads
        # queries: [batch_size * seq_len, n_heads, 1, head_dim]
        # keys/values/relations: [batch_size * seq_len, n_heads, n_neighbor, head_dim]
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        relations = self._split_heads(relations)

        # Compute attention
        # Standard scaled dot-product attention
        # [batch * seq, n_heads, 1, n_neighbor]
        matmul_qk = tf.matmul(queries, keys, transpose_b=True)

        # Scale
        dk = tf.cast(self.head_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)

        # Add relation bias
        # [batch * seq, n_heads, 1, head_dim] x [batch * seq, n_heads, n_neighbor, head_dim]^T
        relation_bias = tf.matmul(queries, relations, transpose_b=True)
        relation_bias = relation_bias / tf.sqrt(dk)

        # Add user preference bias
        user_emb_expanded = tf.reshape(
            user_embeddings, [batch_size, 1, self.dim])
        user_emb_expanded = tf.tile(
            user_emb_expanded, [1, seq_len, 1])
        user_emb_expanded = tf.reshape(
            user_emb_expanded, [batch_size * seq_len, 1, self.dim])

        # Split user embeddings into heads
        user_emb_heads = self._split_heads(user_emb_expanded)

        # Compute user bias: [batch * seq, n_heads, 1, n_neighbor]
        user_bias = tf.matmul(user_emb_heads, keys, transpose_b=True)
        user_bias = user_bias / tf.sqrt(dk)

        # Combine all attention signals
        attention_logits = scaled_attention_logits + \
            0.5 * relation_bias + 0.3 * user_bias

        # Softmax: [batch * seq, n_heads, 1, n_neighbor]
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)

        # Apply dropout to attention weights
        attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout)

        # Apply attention to values
        # [batch * seq, n_heads, 1, head_dim]
        attention_output = tf.matmul(attention_weights, values)

        # Merge heads: [batch * seq, 1, dim]
        attention_output = self._merge_heads(attention_output)

        # Reshape back to [batch_size, seq_len, dim]
        attention_output = tf.reshape(
            attention_output, [batch_size, seq_len, self.dim])

        # Output projection
        output = tf.reshape(attention_output, [-1, self.dim])
        output = tf.matmul(output, self.W_output) + self.bias_output
        output = tf.reshape(output, [batch_size, seq_len, self.dim])

        # Add residual connection
        output = output + self_vectors

        # Layer normalization
        mean, variance = tf.nn.moments(output, axes=[-1], keepdims=True)
        output = (output - mean) / tf.sqrt(variance + 1e-6)
        output = output * self.gamma + self.beta

        # Apply activation
        if self.act is not None:
            output = self.act(output)

        return output
