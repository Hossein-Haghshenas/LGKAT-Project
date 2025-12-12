import tensorflow as tf
import numpy as np
import scipy.sparse as sp


class DiffusionMatrixEnricher:
    """
    Probabilistic Diffusion Model for enriching sparse user-item interaction matrices.

    Based on DDPM (Denoising Diffusion Probabilistic Models) adapted for 
    recommendation systems. This module learns to denoise and reconstruct 
    interaction patterns, filling in missing edges with learned probabilities.

    Architecture:
    - Forward process: Gradually adds Gaussian noise to interaction matrix
    - Reverse process: Learns to denoise and predict missing interactions
    - Uses multi-layer denoising network with temporal embeddings
    """

    def __init__(self,
                 n_users,
                 n_items,
                 embedding_dim=64,
                 num_timesteps=100,
                 beta_start=1e-4,
                 beta_end=0.02,
                 hidden_dims=None,
                 dropout_rate=0.1,
                 prediction_type='epsilon',
                 enrichment_threshold=0.3,
                 top_k_fill=50,
                 name='diffusion_matrix_enricher'):
        """
        Initialize the Diffusion Matrix Enricher.

        Args:
            n_users: Number of users in the system
            n_items: Number of items in the system
            embedding_dim: Dimension for user/item embeddings
            num_timesteps: Number of diffusion timesteps (T)
            beta_start: Starting noise level
            beta_end: Ending noise level
            hidden_dims: Hidden layer dimensions for denoising network
            dropout_rate: Dropout rate for regularization
            prediction_type: 'epsilon' (predict noise) or 'x0' (predict clean matrix)
            enrichment_threshold: Probability threshold for adding new edges
            top_k_fill: Maximum number of edges to add per user
            name: Variable scope name
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.hidden_dims = hidden_dims or [256, 256, 128]
        self.dropout_rate = dropout_rate
        self.prediction_type = prediction_type
        self.enrichment_threshold = enrichment_threshold
        self.top_k_fill = top_k_fill
        self.name = name

        # Compute noise schedule
        self._compute_schedule()

        # Build networks
        self._build_embedding_networks()
        self._build_denoising_network()

        print(f"\n{'='*60}")
        print(f"Diffusion Matrix Enricher Initialized")
        print(f"{'='*60}")
        print(f"Matrix size: {n_users} users × {n_items} items")
        print(f"Embedding dim: {embedding_dim}")
        print(f"Timesteps: {num_timesteps}")
        print(f"Enrichment threshold: {enrichment_threshold}")
        print(f"Top-K fill per user: {top_k_fill}")
        print(f"{'='*60}\n")

    def _compute_schedule(self):
        """Compute noise schedule using cosine schedule (better than linear)."""
        # Cosine schedule (from Improved DDPM)
        steps = np.arange(self.num_timesteps + 1,
                          dtype=np.float32) / self.num_timesteps
        alpha_bars = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]

        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        betas = np.clip(betas, self.beta_start, self.beta_end)

        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)

        # Store as TensorFlow constants
        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas = tf.constant(alphas, dtype=tf.float32)
        self.alpha_bars = tf.constant(alpha_bars, dtype=tf.float32)

        # Precompute values for efficiency
        self.sqrt_alpha_bars = tf.constant(
            np.sqrt(alpha_bars), dtype=tf.float32)
        self.sqrt_one_minus_alpha_bars = tf.constant(
            np.sqrt(1.0 - alpha_bars), dtype=tf.float32)
        self.sqrt_recip_alphas = tf.constant(
            np.sqrt(1.0 / alphas), dtype=tf.float32)

        # Posterior variance
        alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])
        posterior_variance = betas * \
            (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.posterior_variance = tf.constant(
            posterior_variance, dtype=tf.float32)

    def _build_embedding_networks(self):
        """Build user and item embedding networks."""
        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            # User embeddings
            self.user_embeddings = tf.compat.v1.get_variable(
                name='user_embeddings',
                shape=[self.n_users, self.embedding_dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform"
                )
            )

            # Item embeddings
            self.item_embeddings = tf.compat.v1.get_variable(
                name='item_embeddings',
                shape=[self.n_items, self.embedding_dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform"
                )
            )

    def _build_denoising_network(self):
        """Build the denoising U-Net style network."""
        self.weights = {}
        self.biases = {}

        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            # Time embedding dimension
            self.time_embed_dim = self.embedding_dim

            # Input: interaction vector + time embedding
            input_dim = self.n_items + self.time_embed_dim

            # Build encoder (downsampling path)
            prev_dim = input_dim
            for i, hidden_dim in enumerate(self.hidden_dims):
                self.weights[f'enc_w_{i}'] = tf.compat.v1.get_variable(
                    name=f'encoder_weight_{i}',
                    shape=[prev_dim, hidden_dim],
                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                        scale=1.0, mode="fan_avg", distribution="uniform"
                    )
                )
                self.biases[f'enc_b_{i}'] = tf.compat.v1.get_variable(
                    name=f'encoder_bias_{i}',
                    shape=[hidden_dim],
                    initializer=tf.compat.v1.zeros_initializer()
                )
                prev_dim = hidden_dim

            # Build decoder (upsampling path with skip connections)
            decoder_dims = list(
                reversed(self.hidden_dims[:-1])) + [self.n_items]
            for i, hidden_dim in enumerate(decoder_dims):
                # Skip connection doubles input
                skip_dim = prev_dim + \
                    self.hidden_dims[len(
                        self.hidden_dims) - i - 2] if i < len(self.hidden_dims) - 1 else prev_dim

                self.weights[f'dec_w_{i}'] = tf.compat.v1.get_variable(
                    name=f'decoder_weight_{i}',
                    shape=[skip_dim, hidden_dim],
                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                        scale=1.0, mode="fan_avg", distribution="uniform"
                    )
                )
                self.biases[f'dec_b_{i}'] = tf.compat.v1.get_variable(
                    name=f'decoder_bias_{i}',
                    shape=[hidden_dim],
                    initializer=tf.compat.v1.zeros_initializer()
                )
                prev_dim = hidden_dim

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """Create sinusoidal positional embeddings for timesteps."""
        timesteps = tf.cast(timesteps, tf.float32)
        half_dim = embedding_dim // 2
        emb_scale = np.log(10000.0) / (half_dim - 1)

        freqs = tf.exp(-emb_scale * tf.cast(tf.range(half_dim), tf.float32))
        args = tf.expand_dims(timesteps, -1) * tf.expand_dims(freqs, 0)

        embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)

        if embedding_dim % 2 == 1:
            embedding = tf.concat(
                [embedding, tf.zeros_like(embedding[:, :1])], axis=-1)

        return embedding

    def _denoise_network(self, noisy_interactions, timesteps, is_training=True):
        """
        Denoising network that predicts noise or clean interactions.

        Args:
            noisy_interactions: [batch_size, n_items] noisy interaction vectors
            timesteps: [batch_size] timestep indices
            is_training: Whether in training mode

        Returns:
            prediction: [batch_size, n_items] predicted noise or clean interactions
        """
        # Get time embeddings
        time_emb = self._get_timestep_embedding(timesteps, self.time_embed_dim)

        # Concatenate noisy interactions with time embeddings
        x = tf.concat([noisy_interactions, time_emb], axis=-1)

        # Encoder path with skip connections storage
        skip_connections = []
        for i in range(len(self.hidden_dims)):
            x = tf.matmul(
                x, self.weights[f'enc_w_{i}']) + self.biases[f'enc_b_{i}']
            x = tf.nn.leaky_relu(x)

            if is_training:
                x = tf.nn.dropout(x, rate=self.dropout_rate)

            skip_connections.append(x)

        # Decoder path with skip connections
        decoder_dims = list(reversed(self.hidden_dims[:-1])) + [self.n_items]
        for i in range(len(decoder_dims)):
            # Add skip connection (except for output layer)
            if i < len(self.hidden_dims) - 1:
                skip_idx = len(self.hidden_dims) - i - 2
                x = tf.concat([x, skip_connections[skip_idx]], axis=-1)

            x = tf.matmul(
                x, self.weights[f'dec_w_{i}']) + self.biases[f'dec_b_{i}']

            # No activation on final layer
            if i < len(decoder_dims) - 1:
                x = tf.nn.leaky_relu(x)
                if is_training:
                    x = tf.nn.dropout(x, rate=self.dropout_rate)

        return x

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: Add noise to interaction vectors.

        Args:
            x_0: [batch_size, n_items] clean interaction vectors
            t: [batch_size] timestep indices
            noise: Optional pre-generated noise

        Returns:
            x_t: Noisy interaction vectors
            noise: The noise that was added
        """
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_0))

        # Gather schedule values
        sqrt_alpha_bar_t = tf.expand_dims(
            tf.gather(self.sqrt_alpha_bars, t), -1)
        sqrt_one_minus_alpha_bar_t = tf.expand_dims(
            tf.gather(self.sqrt_one_minus_alpha_bars, t), -1
        )

        # Apply forward diffusion
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def p_sample(self, x_t, t, is_training=True):
        """
        Reverse diffusion: Denoise interaction vectors.

        Args:
            x_t: [batch_size, n_items] noisy interaction vectors
            t: [batch_size] timestep indices
            is_training: Whether in training mode

        Returns:
            x_denoised: Denoised interaction vectors
            prediction: Raw network prediction
        """
        prediction = self._denoise_network(x_t, t, is_training)

        if self.prediction_type == 'epsilon':
            # Network predicts noise
            sqrt_alpha_bar_t = tf.expand_dims(
                tf.gather(self.sqrt_alpha_bars, t), -1)
            sqrt_one_minus_alpha_bar_t = tf.expand_dims(
                tf.gather(self.sqrt_one_minus_alpha_bars, t), -1
            )
            x_denoised = (x_t - sqrt_one_minus_alpha_bar_t *
                          prediction) / (sqrt_alpha_bar_t + 1e-8)
        else:
            # Network directly predicts clean interactions
            x_denoised = prediction

        return x_denoised, prediction

    def compute_loss(self, x_0, t=None, noise=None):
        """
        Compute diffusion training loss.

        Args:
            x_0: [batch_size, n_items] clean interaction vectors
            t: Optional timestep indices
            noise: Optional pre-generated noise

        Returns:
            loss: Diffusion loss (MSE)
            x_recon: Reconstructed interaction vectors
        """
        batch_size = tf.shape(x_0)[0]

        if t is None:
            t = tf.random.uniform(
                shape=[batch_size],
                minval=0,
                maxval=self.num_timesteps,
                dtype=tf.int32
            )

        # Forward process
        x_t, actual_noise = self.q_sample(x_0, t, noise)

        # Reverse process
        x_recon, prediction = self.p_sample(x_t, t, is_training=True)

        # Compute loss
        if self.prediction_type == 'epsilon':
            loss = tf.reduce_mean(tf.square(prediction - actual_noise))
        else:
            loss = tf.reduce_mean(tf.square(prediction - x_0))

        return loss, x_recon

    def progressive_denoise(self, x_t, num_steps=None):
        """
        Progressively denoise from pure noise to clean interactions.

        Args:
            x_t: [batch_size, n_items] starting noisy vectors (or pure noise)
            num_steps: Number of denoising steps (defaults to num_timesteps)

        Returns:
            x_0: [batch_size, n_items] denoised interaction vectors
        """
        if num_steps is None:
            num_steps = self.num_timesteps

        batch_size = tf.shape(x_t)[0]

        # Start from timestep T and work backwards
        for t_idx in reversed(range(num_steps)):
            t = tf.ones([batch_size], dtype=tf.int32) * t_idx

            # Denoise one step
            x_denoised, _ = self.p_sample(x_t, t, is_training=False)

            if t_idx > 0:
                # Add noise for next step (except at t=0)
                alpha_t = tf.expand_dims(tf.gather(self.alphas, t), -1)
                noise = tf.random.normal(shape=tf.shape(x_t))
                x_t = x_denoised * tf.sqrt(alpha_t) + \
                    noise * tf.sqrt(1 - alpha_t)
            else:
                x_t = x_denoised

        return x_t

    def enrich_interaction_matrix(self, sess, sparse_matrix, batch_size=512):
        """
        Main enrichment function: Takes sparse matrix, returns enriched dense matrix.

        Args:
            sess: TensorFlow session
            sparse_matrix: scipy.sparse matrix [n_users, n_items] with existing interactions
            batch_size: Batch size for processing

        Returns:
            enriched_matrix: scipy.sparse matrix with added predicted interactions
            enrichment_stats: Dictionary with statistics about enrichment
        """
        print(f"\n{'='*60}")
        print("Starting Matrix Enrichment Process")
        print(f"{'='*60}")

        # Convert to dense for processing
        dense_matrix = sparse_matrix.toarray().astype(np.float32)
        original_edges = np.sum(dense_matrix > 0)

        print(f"Original matrix: {self.n_users} × {self.n_items}")
        print(f"Original edges: {original_edges:,}")
        print(
            f"Sparsity: {100 * (1 - original_edges / (self.n_users * self.n_items)):.4f}%")

        enriched_matrix = dense_matrix.copy()
        new_edges = 0

        # Process in batches
        for start_idx in range(0, self.n_users, batch_size):
            end_idx = min(start_idx + batch_size, self.n_users)
            batch_users = dense_matrix[start_idx:end_idx]
            curr_batch_size = end_idx - start_idx

            # Start with small noise (not pure noise) for stability
            t = tf.ones([curr_batch_size], dtype=tf.int32) * \
                (self.num_timesteps // 4)
            x_t, _ = self.q_sample(
                tf.constant(batch_users, dtype=tf.float32),
                t
            )

            # Denoise to get predictions
            x_pred = self.progressive_denoise(
                x_t, num_steps=self.num_timesteps // 2)
            x_pred_np = sess.run(x_pred)

            # Apply sigmoid to get probabilities
            probs = 1 / (1 + np.exp(-x_pred_np))

            # For each user, add top-K predicted interactions above threshold
            for i, user_idx in enumerate(range(start_idx, end_idx)):
                existing_items = dense_matrix[user_idx] > 0

                # Mask out existing interactions
                probs[i][existing_items] = 0

                # Get top-K predictions
                top_k_items = np.argsort(probs[i])[-self.top_k_fill:]

                # Add edges above threshold
                for item_idx in top_k_items:
                    if probs[i, item_idx] >= self.enrichment_threshold:
                        enriched_matrix[user_idx,
                                        item_idx] = probs[i, item_idx]
                        new_edges += 1

            if (start_idx // batch_size + 1) % 10 == 0:
                print(f"Processed {end_idx}/{self.n_users} users...")

        final_edges = np.sum(enriched_matrix > 0)

        stats = {
            'original_edges': int(original_edges),
            'new_edges': int(new_edges),
            'final_edges': int(final_edges),
            'enrichment_rate': float(new_edges / original_edges),
            'final_sparsity': float(1 - final_edges / (self.n_users * self.n_items))
        }

        print(f"\n{'='*60}")
        print("Enrichment Complete")
        print(f"{'='*60}")
        print(f"New edges added: {new_edges:,}")
        print(f"Final edges: {final_edges:,}")
        print(f"Enrichment rate: {stats['enrichment_rate']:.2%}")
        print(f"Final sparsity: {100 * stats['final_sparsity']:.4f}%")
        print(f"{'='*60}\n")

        # Convert back to sparse
        enriched_sparse = sp.csr_matrix(enriched_matrix)

        return enriched_sparse, stats

    def get_training_ops(self, learning_rate=1e-4):
        """
        Create training operations for the diffusion model.

        Args:
            learning_rate: Learning rate for optimizer

        Returns:
            Dictionary with training operations
        """
        # Placeholders
        self.interaction_input = tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=[None, self.n_items],
            name='interaction_input'
        )

        self.timestep_input = tf.compat.v1.placeholder(
            dtype=tf.int32,
            shape=[None],
            name='timestep_input'
        )

        # Compute loss
        self.diffusion_loss, self.reconstructed = self.compute_loss(
            self.interaction_input,
            self.timestep_input
        )

        # Add L2 regularization
        self.l2_loss = tf.constant(0.0)
        for key in self.weights:
            self.l2_loss += tf.nn.l2_loss(self.weights[key])

        # Total loss
        self.total_loss = self.diffusion_loss + 1e-5 * self.l2_loss

        # Optimizer
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate).minimize(self.total_loss)

        return {
            'loss': self.total_loss,
            'diffusion_loss': self.diffusion_loss,
            'optimizer': self.optimizer,
            'reconstructed': self.reconstructed
        }

    def train_on_batch(self, sess, user_interactions, ops):
        """
        Train the diffusion model on a batch of user interactions.

        Args:
            sess: TensorFlow session
            user_interactions: [batch_size, n_items] interaction vectors
            ops: Dictionary of training operations

        Returns:
            loss: Training loss value
        """
        batch_size = user_interactions.shape[0]

        # Sample random timesteps
        t = np.random.randint(0, self.num_timesteps, size=batch_size)

        feed_dict = {
            self.interaction_input: user_interactions,
            self.timestep_input: t
        }

        _, loss, diff_loss = sess.run(
            [ops['optimizer'], ops['loss'], ops['diffusion_loss']],
            feed_dict=feed_dict
        )

        return loss, diff_loss
