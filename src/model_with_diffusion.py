import tensorflow as tf
import numpy as np
from latent_diffusion import LatentDiffusion, LatentDiffusionPreprocessor


def integrate_diffusion_with_kgcn(model_class):
    """
    example showing how to integrate latent diffusion with existing kgcn model.
    this is a mixin pattern that can be adapted to your specific model.
    """

    class KGCNWithDiffusion(model_class):
        """
        extended kgcn model with latent diffusion preprocessing.
        """

        def _parse_args(self, args, n_user, n_item, adj_entity, adj_relation, user_item_adj):
            # call parent initialization
            super()._parse_args(args, n_user, n_item, adj_entity, adj_relation, user_item_adj)

            # diffusion hyperparameters (can be added to args)
            self.use_diffusion = getattr(args, 'use_diffusion', False)
            self.diffusion_weight = getattr(args, 'diffusion_weight', 0.1)
            self.diffusion_timesteps = getattr(
                args, 'diffusion_timesteps', 1000)
            self.diffusion_beta_start = getattr(
                args, 'diffusion_beta_start', 1e-4)
            self.diffusion_beta_end = getattr(args, 'diffusion_beta_end', 0.02)

            # create diffusion preprocessor if enabled
            if self.use_diffusion:
                hidden_dims = [self.dim * 2, self.dim * 2, self.dim]
                self.diffusion_preprocessor = LatentDiffusionPreprocessor(
                    user_embedding_dim=self.dim,
                    item_embedding_dim=self.dim,
                    num_timesteps=self.diffusion_timesteps,
                    beta_start=self.diffusion_beta_start,
                    beta_end=self.diffusion_beta_end,
                    hidden_dims=hidden_dims,
                    diffusion_weight=self.diffusion_weight,
                    apply_to_users=True,
                    apply_to_items=True,
                    name='diffusion_preprocessor'
                )

                print(f"\n--------------- Diffusion Configuration ---------------")
                print(f"Diffusion enabled: {self.use_diffusion}")
                print(f"Diffusion weight: {self.diffusion_weight}")
                print(f"Timesteps: {self.diffusion_timesteps}")
                print(
                    f"Beta range: [{self.diffusion_beta_start}, {self.diffusion_beta_end}]")
                print(f"-------------------------------------------------------\n")

        def _build_model(self, n_user, n_item, n_entity, n_relation):
            # call parent model building
            super()._build_model(n_user, n_item, n_entity, n_relation)

            # apply diffusion preprocessing if enabled
            if self.use_diffusion:
                # preprocess embeddings through diffusion
                self.user_embeddings_diffused, self.item_embeddings_diffused, self.diffusion_loss = \
                    self.diffusion_preprocessor.preprocess_embeddings(
                        user_embeddings=self.user_embeddings_final,
                        item_embeddings=self.item_embeddings_final,
                        is_training=True
                    )

                # use diffused embeddings for scoring
                self.scores = tf.reduce_sum(
                    input_tensor=self.user_embeddings_diffused * self.item_embeddings_diffused,
                    axis=1
                )
                self.scores_normalized = tf.sigmoid(self.scores)
            else:
                self.diffusion_loss = tf.constant(0.0)

        def _build_train(self):
            # call parent training setup
            super()._build_train()

            # add diffusion loss if enabled
            if self.use_diffusion:
                self.loss = self.loss + self.diffusion_loss

                # add diffusion weights to l2 regularization
                self.l2_loss = self.l2_loss + \
                    self.diffusion_preprocessor.get_regularization_loss()

        def train(self, sess, feed_dict):
            if self.use_diffusion:
                return sess.run(
                    [self.optimizer, self.base_loss,
                        self.cl_loss, self.diffusion_loss],
                    feed_dict
                )
            else:
                return sess.run([self.optimizer, self.base_loss, self.cl_loss], feed_dict)

    return KGCNWithDiffusion


# standalone usage example
def example_standalone_usage():
    """
    example of using latent diffusion as a standalone preprocessing module.
    """
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    # configuration
    batch_size = 256
    embedding_dim = 64

    # create placeholder for input embeddings (from existing model)
    input_embeddings = tf.compat.v1.placeholder(
        dtype=tf.float32,
        shape=[None, embedding_dim],
        name='input_embeddings'
    )

    # create diffusion module
    diffusion = LatentDiffusion(
        embedding_dim=embedding_dim,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        hidden_dims=[128, 128, 64],
        activation='leaky_relu',
        dropout_rate=0.1,
        prediction_type='epsilon',
        name='embedding_diffusion'
    )

    # preprocess embeddings
    processed_embeddings, diffusion_loss = diffusion.preprocess(
        input_embeddings,
        is_training=True
    )

    # the processed_embeddings can now be used in your main model
    print("Input shape:", input_embeddings.shape)
    print("Output shape:", processed_embeddings.shape)
    print("Diffusion loss:", diffusion_loss)

    # example training loop integration
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # simulate input embeddings
        dummy_embeddings = np.random.randn(
            batch_size, embedding_dim).astype(np.float32)

        # run preprocessing
        processed, loss = sess.run(
            [processed_embeddings, diffusion_loss],
            feed_dict={input_embeddings: dummy_embeddings}
        )

        print(f"Processed embeddings shape: {processed.shape}")
        print(f"Diffusion loss value: {loss:.4f}")


if __name__ == '__main__':
    example_standalone_usage()
