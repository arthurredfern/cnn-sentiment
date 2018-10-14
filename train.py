import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from data import make_dataset
from model import CNNClassifier

def main():
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')

    with tf.name_scope('data'):
        train_data = make_dataset('data/clean/train.tfrecord', batch_size=64)
        dev_data = make_dataset('data/clean/dev.tfrecord', batch_size=128)

        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)
        lengths, labels, words = iterator.get_next()
        train_iterator_init = iterator.make_initializer(train_data)
        dev_iterator_init = iterator.make_initializer(dev_data)

    # Model
    #emb_array = np.zeros((400002, 10))
    emb_array = np.load('embeddings/glove.6B.300d.npy')
    filter_sizes = [3, 4, 5]
    model = CNNClassifier(filter_sizes)
    dropout_rate = tf.placeholder_with_default(0.0, shape=[])
    logits = model.build_graph(words, emb_array, dropout_rate)

    with tf.name_scope('loss'):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                       logits=logits)
        loss = tf.reduce_mean(loss)
        loss_summary = tf.summary.scalar('loss', loss)
    objective = tf.train.AdamOptimizer().minimize(loss)

    with tf.name_scope('accuracy'):
        prediction = tf.nn.sigmoid(logits) > 0.5
        correct = tf.equal(tf.cast(prediction, tf.int32), tf.cast(labels, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    logdir = os.path.join('runs', now)
    writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10
    steps = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            print('Epoch {:d}'.format(epoch))
            sess.run(train_iterator_init)
            try:
                while True:
                    _, loss_val, acc_val, loss_summ = sess.run([objective,
                        loss, accuracy, loss_summary],
                        feed_dict={dropout_rate: 0.5})
                    steps += 1
                    writer.add_summary(loss_summ, global_step=steps)
                    print('\rloss: {:.4f} accuracy: {:.4f}'.format(
                        loss_val, acc_val), end='', flush=True)
            except tf.errors.OutOfRangeError:
                pass

            sess.run(dev_iterator_init)
            avg_accuracy = 0
            count = 0
            try:
                while True:
                    acc = sess.run(accuracy)
                    avg_accuracy += acc
                    count += 1
            except tf.errors.OutOfRangeError:
                pass
            avg_accuracy /= count
            print('\nDev accuracy: {:.4f}'.format(avg_accuracy))
            acc_summ = tf.Summary()
            acc_summ.value.add(tag='dev/accuracy', simple_value=avg_accuracy)
            writer.add_summary(acc_summ, global_step=steps)

    writer.close()

if __name__ == '__main__':
    main()
