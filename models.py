from collections import OrderedDict
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, features_info, emb_dim, name_prefix=''):
        """
        sequence対応のembedding layer
        """
        super(EmbeddingLayer, self).__init__()
        self.features_info = features_info
        self.feature_to_embedding_layer = OrderedDict()
        for feature in features_info:
            initializer = tf.keras.initializers.RandomNormal(stddev=0.01, seed=None)
            if feature['is_sequence']:
                # sequenceのembedding
                self.feature_to_embedding_layer[feature['name']] = tf.keras.layers.Embedding(
                    feature['dim'],
                    emb_dim,
                    mask_zero=True,
                    name=f"embedding_{name_prefix}{feature['name']}",
                    embeddings_initializer=initializer)
            else:
                self.feature_to_embedding_layer[feature['name']] = tf.keras.layers.Embedding(
                    feature['dim'],
                    emb_dim,
                    name=f"embedding_{name_prefix}{feature['name']}",
                    embeddings_initializer=initializer)

    def concatenate_embeddings(self, embeddings, name_prefix=''):
        if len(embeddings) >= 2:
            embeddings = tf.keras.layers.Concatenate(axis=1, name=name_prefix+'embeddings_concat')(embeddings)
        else:
            embeddings = embeddings[0]
        return embeddings

    def call(self, inputs):
        embeddings = []
        for feature_input, feature in zip(inputs, self.features_info):
            # embeddingの作成
            embedding = self.feature_to_embedding_layer[feature['name']](feature_input)
            if feature['is_sequence']:
                # sequenceの場合はaverage pooling
                embedding = tf.math.reduce_mean(embedding, axis=1, keepdims=True)
            embeddings.append(embedding)

        # concatenate
        embeddings = self.concatenate_embeddings(embeddings)
        return embeddings


class FactorizeLayer(tf.keras.layers.Layer):
    def __init__(self, features_info, latent_dim=5):
        super(FactorizeLayer, self).__init__()
        self.embedding = EmbeddingLayer(features_info, latent_dim, 'factorize_')

    def call(self, inputs):
        # factorization
        embeddings = self.embedding(inputs)
        # 元論文のlemma 3.1
        summed_square = tf.square(tf.reduce_sum(embeddings, axis=1))
        squared_sum = tf.reduce_sum(tf.square(embeddings), axis=1)
        output = tf.subtract(summed_square, squared_sum)
        output = tf.multiply(0.5, tf.reduce_sum(output, axis=1, keepdims=True))

        return output


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, features_info):
        super(LinearLayer, self).__init__()
        self.linear_embedding = EmbeddingLayer(features_info, 1, 'linear_')
        self.linear_layer = tf.keras.layers.Dense(1, activation='relu', name='linear_dense')

    def call(self, inputs):
        embeddings = self.linear_embedding(inputs)
        # reduce_sum → bias(ones_like)の方が正しいかも
        embeddings = tf.squeeze(embeddings, axis=2)
        output = self.linear_layer(embeddings)
        return output


class FactorizationMachines(tf.keras.Model):
    def __init__(self, features_info, latent_dim=5):
        super(FactorizationMachines, self).__init__()
        self.factorize_layer = FactorizeLayer(features_info, latent_dim=latent_dim)
        self.linear_layer = LinearLayer(features_info)

    def call(self, inputs):
        linear_terms = self.linear_layer(inputs)
        factorization_terms = self.factorize_layer(inputs)
        output = tf.add(linear_terms, factorization_terms)

        return tf.keras.activations.sigmoid(output)
