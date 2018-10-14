import tensorflow as tf

class CNNClassifier:
    """A CNN for sentence classification, based on
    Kim, Y. (2014). Convolutional neural networks for sentence classification.
    Args:
        filter_sizes: list, contains size of filters (int) used in the
            convolutional layer.
        dropout_rate: float, dropout rate used before the last dense layer.
    """
    def __init__(self, filter_sizes):
        self.filter_sizes = filter_sizes

    def build_graph(self, inputs, emb_array, dropout_rate):
        """Build the graph of the model.
        Args:
            inputs: tensor, mini-batch of word IDs for sentences
            emb_array: numpy array of shape (vocabulary_size, emb_dimension)
                used to initialize the embedding layer.
        Returns:
            logit, the output tensor of the classifier.
        """
        # Embedding layer
        embeddings = tf.get_variable('embeddings', shape=emb_array.shape,
                        initializer=tf.constant_initializer(emb_array))
        x = tf.nn.embedding_lookup(embeddings, inputs)

        # Features from convolutional layers, one per filter size
        features = []
        for i, filter_size in enumerate(self.filter_sizes):
            # Convolution
            f = tf.layers.conv1d(x, filters=100, kernel_size=3,
                                           padding='same', activation=tf.nn.relu,
                                  name='conv{}'.format(i + 1))
            # Max-pooling over time
            with tf.name_scope('max-time-pool{}'.format(i + 1)):
                features.append(tf.reduce_max(f, axis=1))

        with tf.name_scope('concat'):
            x = tf.concat(features, axis=1)

        x = tf.layers.dropout(x, rate=dropout_rate)
        logit = tf.layers.dense(x, 1)

        return logit
