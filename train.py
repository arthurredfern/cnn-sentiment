import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from data import make_dataset
from model import CNNClassifier

now = datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')

with tf.name_scope('data'):
    train_data = make_dataset('data/clean/train.tfrecord', batch_size=64)
    iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                               train_data.output_shapes)
    lengths, labels, words = iterator.get_next()
    train_iterator_init = iterator.make_initializer(train_data)

# Model
emb_array = np.load('embeddings/glove.6B.300d.npy')
filter_sizes = [3, 4, 5]
model = CNNClassifier(filter_sizes, dropout_rate=0.5)
logits = model.build_graph(words, emb_array)

with tf.name_scope('loss'):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                   logits=logits)
    loss = tf.reduce_mean(loss)
    loss_summary = tf.summary.scalar('loss', loss)
objective = tf.train.AdamOptimizer().minimize(loss)

logdir = os.path.join('runs', now)
writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
steps = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        sess.run(train_iterator_init)

        try:
            while True:
                _, loss_sum = sess.run([objective, loss_summary])
                steps += 1
                writer.add_summary(loss_sum, global_step=steps)
                print(steps)
        except tf.errors.OutOfRangeError:
            pass

writer.close()
