import tensorflow as tf
import numpy as np


class LatentDiffusion:
    """
    latent diffusion module for preprocessing embeddings in recommendation systems.
    implements forward diffusion (q_sample) and reverse diffusion (p_sample) processes.
    """

    def __init__(self,
                 embedding_dim,
                 num_timesteps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 hidden_dims=None,
                 activation='leaky_relu',
                 dropout_rate=0.1,
                 prediction_type='epsilon',
                 name='latent_diffusion'):
        """
        initialize the latent diffusion module.

        args:
            embedding_dim: dimension of input embeddings
            num_timesteps: number of diffusion timesteps (T)
            beta_start: starting value for noise schedule
            beta_end: ending value for noise schedule
            hidden_dims: list of hidden layer dimensions for denoising mlp
            activation: activation function ('leaky_relu', 'relu', 'gelu')
            dropout_rate: dropout rate for mlp layers
            prediction_type: 'epsilon' to predict noise, 'x0' to predict clean embedding
            name: variable scope name
        """
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.hidden_dims = hidden_dims or [
            embedding_dim * 2, embedding_dim * 2, embedding_dim]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.prediction_type = prediction_type
        self.name = name

        # compute noise schedule parameters
        self._compute_schedule()

        # build denoising network
        self._build_network()

    def _compute_schedule(self):
        """
        compute the noise schedule (betas) and derived quantities (alphas, alpha_bars).
        uses linear schedule by default, can be extended to cosine schedule.
        """
        # linear beta schedule
        betas = np.linspace(self.beta_start, self.beta_end,
                            self.num_timesteps, dtype=np.float32)

        # compute alphas
        alphas = 1.0 - betas

        # compute cumulative product of alphas (alpha_bar)
        alpha_bars = np.cumprod(alphas)

        # store as tensorflow constants for efficient gpu computation
        self.betas = tf.constant(betas, dtype=tf.float32)
        self.alphas = tf.constant(alphas, dtype=tf.float32)
        self.alpha_bars = tf.constant(alpha_bars, dtype=tf.float32)

        # precompute sqrt values for forward process
        self.sqrt_alpha_bars = tf.constant(
            np.sqrt(alpha_bars), dtype=tf.float32)
        self.sqrt_one_minus_alpha_bars = tf.constant(
            np.sqrt(1.0 - alpha_bars), dtype=tf.float32)

        # precompute values for reverse process
        self.sqrt_recip_alphas = tf.constant(
            np.sqrt(1.0 / alphas), dtype=tf.float32)

        # posterior variance for reverse process
        alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])
        posterior_variance = betas * \
            (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.posterior_variance = tf.constant(
            posterior_variance, dtype=tf.float32)

    def _get_activation(self):
        """
        return the activation function based on configuration.
        """
        if self.activation == 'leaky_relu':
            return tf.nn.leaky_relu
        elif self.activation == 'relu':
            return tf.nn.relu
        elif self.activation == 'gelu':
            return lambda x: x * tf.sigmoid(1.702 * x)  # approximate gelu
        else:
            return tf.nn.leaky_relu

    def _build_network(self):
        """
        build the denoising mlp network.
        the network takes noisy embeddings and timestep embeddings as input
        and predicts either the noise or the clean embedding.
        """
        self.weights = {}
        self.biases = {}

        activation_fn = self._get_activation()

        with tf.compat.v1.variable_scope(self.name, reuse=tf.compat.v1.AUTO_REUSE):
            # timestep embedding layer (sinusoidal positional encoding)
            self.time_embed_dim = self.embedding_dim

            # input projection: concatenate embedding and time embedding
            input_dim = self.embedding_dim + self.time_embed_dim

            # build mlp layers
            prev_dim = input_dim
            for i, hidden_dim in enumerate(self.hidden_dims):
                self.weights[f'layer_{i}'] = tf.compat.v1.get_variable(
                    name=f'weight_{i}',
                    shape=[prev_dim, hidden_dim],
                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                        scale=1.0, mode="fan_avg", distribution="uniform"
                    )
                )
                self.biases[f'layer_{i}'] = tf.compat.v1.get_variable(
                    name=f'bias_{i}',
                    shape=[hidden_dim],
                    initializer=tf.compat.v1.zeros_initializer()
                )
                prev_dim = hidden_dim

            # output projection layer
            self.weights['output'] = tf.compat.v1.get_variable(
                name='weight_output',
                shape=[prev_dim, self.embedding_dim],
                initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                    scale=1.0, mode="fan_avg", distribution="uniform"
                )
            )
            self.biases['output'] = tf.compat.v1.get_variable(
                name='bias_output',
                shape=[self.embedding_dim],
                initializer=tf.compat.v1.zeros_initializer()
            )

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """
        create sinusoidal timestep embeddings.

        args:
            timesteps: tensor of shape [batch_size] containing timestep indices
            embedding_dim: dimension of the output embedding

        returns:
            timestep embeddings of shape [batch_size, embedding_dim]
        """
        # ensure timesteps is float for computation
        timesteps = tf.cast(timesteps, tf.float32)

        half_dim = embedding_dim // 2
        emb_scale = np.log(10000.0) / (half_dim - 1)
        emb_scale = tf.constant(emb_scale, dtype=tf.float32)

        # create frequency bands
        freqs = tf.exp(-emb_scale * tf.cast(tf.range(half_dim), tf.float32))

        # compute embeddings
        args = tf.expand_dims(timesteps, -1) * tf.expand_dims(freqs, 0)

        # concatenate sin and cos embeddings
        embedding = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)

        # handle odd embedding dimensions
        if embedding_dim % 2 == 1:
            embedding = tf.concat(
                [embedding, tf.zeros_like(embedding[:, :1])], axis=-1)

        return embedding

    def _gather_with_timesteps(self, values, timesteps):
        """
        gather values at specific timesteps.

        args:
            values: tensor of shape [num_timesteps]
            timesteps: tensor of shape [batch_size] with timestep indices

        returns:
            gathered values of shape [batch_size, 1]
        """
        gathered = tf.gather(values, timesteps)
        return tf.expand_dims(gathered, -1)

    def q_sample(self, e_0, t, noise=None):
        """
        forward diffusion process: add noise to embeddings.

        implements: e_t = sqrt(alpha_bar_t) * e_0 + sqrt(1 - alpha_bar_t) * epsilon

        args:
            e_0: clean embeddings of shape [batch_size, embedding_dim]
            t: timesteps of shape [batch_size]
            noise: optional pre-generated noise, if none will generate random noise

        returns:
            e_t: noisy embeddings of shape [batch_size, embedding_dim]
            noise: the noise that was added (useful for computing loss)
        """
        # generate noise if not provided
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(e_0))

        # gather schedule values for the given timesteps
        sqrt_alpha_bar_t = self._gather_with_timesteps(self.sqrt_alpha_bars, t)
        sqrt_one_minus_alpha_bar_t = self._gather_with_timesteps(
            self.sqrt_one_minus_alpha_bars, t)

        # apply forward diffusion formula
        e_t = sqrt_alpha_bar_t * e_0 + sqrt_one_minus_alpha_bar_t * noise

        return e_t, noise

    def _denoise_mlp(self, e_t, t, is_training=True):
        """
        denoising mlp network forward pass.

        args:
            e_t: noisy embeddings of shape [batch_size, embedding_dim]
            t: timesteps of shape [batch_size]
            is_training: whether in training mode (affects dropout)

        returns:
            prediction: predicted noise or clean embedding
        """
        activation_fn = self._get_activation()

        # get timestep embeddings
        time_emb = self._get_timestep_embedding(t, self.time_embed_dim)

        # concatenate noisy embedding with time embedding
        x = tf.concat([e_t, time_emb], axis=-1)

        # forward through mlp layers
        for i in range(len(self.hidden_dims)):
            x = tf.matmul(
                x, self.weights[f'layer_{i}']) + self.biases[f'layer_{i}']
            x = activation_fn(x)

            # apply dropout during training
            if is_training:
                x = tf.nn.dropout(x, rate=self.dropout_rate)

        # output projection
        prediction = tf.matmul(
            x, self.weights['output']) + self.biases['output']

        return prediction

    def p_sample(self, e_t, t, is_training=True):
        """
        reverse diffusion process: denoise embeddings.

        predicts either the noise (epsilon) or the clean embedding (x0)
        based on the prediction_type configuration.

        args:
            e_t: noisy embeddings of shape [batch_size, embedding_dim]
            t: timesteps of shape [batch_size]
            is_training: whether in training mode

        returns:
            e_denoised: denoised embeddings of shape [batch_size, embedding_dim]
            prediction: raw network prediction (noise or x0)
        """
        # get network prediction
        prediction = self._denoise_mlp(e_t, t, is_training)

        if self.prediction_type == 'epsilon':
            # network predicts noise, compute x0 from it
            sqrt_alpha_bar_t = self._gather_with_timesteps(
                self.sqrt_alpha_bars, t)
            sqrt_one_minus_alpha_bar_t = self._gather_with_timesteps(
                self.sqrt_one_minus_alpha_bars, t)

            # rearranging forward equation: x0 = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
            e_denoised = (e_t - sqrt_one_minus_alpha_bar_t *
                          prediction) / (sqrt_alpha_bar_t + 1e-8)
        else:
            # network directly predicts x0
            e_denoised = prediction

        return e_denoised, prediction

    def compute_loss(self, e_0, t=None, noise=None):
        """
        compute diffusion loss for training.

        args:
            e_0: clean embeddings of shape [batch_size, embedding_dim]
            t: optional timesteps, if none will sample randomly
            noise: optional pre-generated noise

        returns:
            loss: mse loss between predicted and actual noise/embedding
            e_gen: generated (denoised) embeddings
        """
        batch_size = tf.shape(e_0)[0]

        # sample random timesteps if not provided
        if t is None:
            t = tf.random.uniform(
                shape=[batch_size],
                minval=0,
                maxval=self.num_timesteps,
                dtype=tf.int32
            )

        # forward diffusion: add noise
        e_t, actual_noise = self.q_sample(e_0, t, noise)

        # reverse diffusion: predict and denoise
        e_gen, prediction = self.p_sample(e_t, t, is_training=True)

        # compute loss
        if self.prediction_type == 'epsilon':
            # loss is mse between predicted and actual noise
            loss = tf.reduce_mean(tf.square(prediction - actual_noise))
        else:
            # loss is mse between predicted and actual clean embedding
            loss = tf.reduce_mean(tf.square(prediction - e_0))

        return loss, e_gen

    def preprocess(self, e_0, t=None, is_training=True):
        """
        main preprocessing pipeline for embeddings.

        takes embeddings from the existing model, applies diffusion-based
        augmentation, and returns processed embeddings.

        args:
            e_0: input embeddings from existing model [batch_size, embedding_dim]
            t: optional specific timestep, if none samples randomly
            is_training: whether in training mode

        returns:
            e_gen: processed embeddings ready for main model
            loss_diffusion: diffusion loss (0 if not training)
        """
        batch_size = tf.shape(e_0)[0]

        if is_training:
            # sample random timesteps during training
            if t is None:
                # use smaller timesteps for less noise during training
                max_t = self.num_timesteps // 4  # use first quarter of schedule
                t = tf.random.uniform(
                    shape=[batch_size],
                    minval=0,
                    maxval=max_t,
                    dtype=tf.int32
                )

            # compute loss and get generated embeddings
            loss_diffusion, e_gen = self.compute_loss(e_0, t)
        else:
            # during inference, use minimal noise for stability
            t = tf.zeros([batch_size], dtype=tf.int32)
            e_t, _ = self.q_sample(e_0, t)
            e_gen, _ = self.p_sample(e_t, t, is_training=False)
            loss_diffusion = tf.constant(0.0)

        return e_gen, loss_diffusion

    def get_regularization_loss(self):
        """
        compute l2 regularization loss for network weights.

        returns:
            l2_loss: sum of l2 norms of all weights
        """
        l2_loss = tf.constant(0.0)
        for key in self.weights:
            l2_loss += tf.nn.l2_loss(self.weights[key])
        return l2_loss


class LatentDiffusionPreprocessor:
    """
    wrapper class for integrating latent diffusion with existing models.
    handles both user and item embeddings with separate diffusion modules.
    """

    def __init__(self,
                 user_embedding_dim,
                 item_embedding_dim=None,
                 num_timesteps=1000,
                 beta_start=1e-4,
                 beta_end=0.02,
                 hidden_dims=None,
                 diffusion_weight=0.1,
                 apply_to_users=True,
                 apply_to_items=True,
                 name='diffusion_preprocessor'):
        """
        initialize the preprocessor.

        args:
            user_embedding_dim: dimension of user embeddings
            item_embedding_dim: dimension of item embeddings (defaults to user_embedding_dim)
            num_timesteps: number of diffusion timesteps
            beta_start: noise schedule start
            beta_end: noise schedule end
            hidden_dims: hidden layer dimensions for mlp
            diffusion_weight: lambda weight for diffusion loss
            apply_to_users: whether to apply diffusion to user embeddings
            apply_to_items: whether to apply diffusion to item embeddings
            name: variable scope name
        """
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim or user_embedding_dim
        self.diffusion_weight = diffusion_weight
        self.apply_to_users = apply_to_users
        self.apply_to_items = apply_to_items
        self.name = name

        # create diffusion modules
        if apply_to_users:
            self.user_diffusion = LatentDiffusion(
                embedding_dim=user_embedding_dim,
                num_timesteps=num_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                hidden_dims=hidden_dims,
                name=f'{name}_user'
            )

        if apply_to_items:
            self.item_diffusion = LatentDiffusion(
                embedding_dim=self.item_embedding_dim,
                num_timesteps=num_timesteps,
                beta_start=beta_start,
                beta_end=beta_end,
                hidden_dims=hidden_dims,
                name=f'{name}_item'
            )

    def preprocess_embeddings(self, user_embeddings=None, item_embeddings=None, is_training=True):
        """
        preprocess user and/or item embeddings through diffusion.

        args:
            user_embeddings: user embeddings tensor [batch_size, user_dim]
            item_embeddings: item embeddings tensor [batch_size, item_dim]
            is_training: whether in training mode

        returns:
            processed_user_emb: processed user embeddings (or original if not applied)
            processed_item_emb: processed item embeddings (or original if not applied)
            total_diffusion_loss: combined diffusion loss
        """
        total_diffusion_loss = tf.constant(0.0)
        processed_user_emb = user_embeddings
        processed_item_emb = item_embeddings

        if self.apply_to_users and user_embeddings is not None:
            processed_user_emb, user_diff_loss = self.user_diffusion.preprocess(
                user_embeddings, is_training=is_training
            )
            total_diffusion_loss += user_diff_loss

        if self.apply_to_items and item_embeddings is not None:
            processed_item_emb, item_diff_loss = self.item_diffusion.preprocess(
                item_embeddings, is_training=is_training
            )
            total_diffusion_loss += item_diff_loss

        # scale by diffusion weight
        total_diffusion_loss = self.diffusion_weight * total_diffusion_loss

        return processed_user_emb, processed_item_emb, total_diffusion_loss

    def get_regularization_loss(self):
        """
        get combined regularization loss from all diffusion modules.
        """
        l2_loss = tf.constant(0.0)
        if self.apply_to_users:
            l2_loss += self.user_diffusion.get_regularization_loss()
        if self.apply_to_items:
            l2_loss += self.item_diffusion.get_regularization_loss()
        return l2_loss
