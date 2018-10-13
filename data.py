"""
A script to read the Customer Review dataset and export cleaned files
with binary labels for sentiment classification.
"""

import os
import re
from os import listdir
from shutil import rmtree
import numpy as np
from collections import defaultdict
import pickle
import tensorflow as tf

data_path = './data'
embeddings_path = './embeddings'
raw_data_path = os.path.join(data_path, 'raw')
clean_data_path = os.path.join(data_path, 'clean')

PAD = '<PAD>'
UNK = '<UNK>'

# Serialization strings
LEN_FEAT_NAME = 'length'
LABEL_FEAT_NAME = 'label'
WORDS_FEAT_NAME = 'words'

# Source: http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# Retrieved on Oct 12, 2018
def make_example(sequence, label):
    """Creates a tf.train.SequenceExample to be written in a TFRecord
    Args:
        sequence (list): contains word IDs (int) in a sequence
        label (int): the label of the sequence
    Returns: a tf.train.SequenceExample
    """
    ex = tf.train.SequenceExample()
    # Context: sequence length and label
    ex.context.feature[LEN_FEAT_NAME].int64_list.value.append(len(sequence))
    ex.context.feature[LABEL_FEAT_NAME].int64_list.value.append(label)

    # Feature lists: words
    fl_tokens = ex.feature_lists.feature_list[WORDS_FEAT_NAME]
    for word in sequence:
        fl_tokens.feature.add().int64_list.value.append(word)

    return ex

def save_to_tfrecord(data, labels, split):
    """Writes a TFRecord containing instances of SequenceExample
    Args:
        data (list): contains lists of word IDs (int), one per example
        labels (list): contains labels (int), one per example
    """
    tfrecord_file = os.path.join(clean_data_path, split + '.tfrecord')
    writer = tf.python_io.TFRecordWriter(tfrecord_file)

    for sequence, label in zip(data, labels):
        example = make_example(sequence, label)
        writer.write(example.SerializeToString())

    writer.close()
    print('Saved TFRecord to {}'.format(tfrecord_file))

def clean_data():
    """Read the raw Customer Review dataset and export a clean file with
    reviews and binary labels.
    """
    print('Cleaning data...')
    files = listdir(raw_data_path)
    files.remove('Readme.txt')

    # Remove any exported clean files if they exist
    if os.path.exists(clean_data_path):
        rmtree(clean_data_path)
    os.mkdir(clean_data_path)

    omit_symbols = ('*', '[t]', '##')

    # To detect one or more annotations like [-1]
    pattern = re.compile('\[[+-]{1}[0-9]\]')
    data = ''
    labels = ''

    for file in files:
        with open(os.path.join(raw_data_path, file)) as in_file:
            for i, line in enumerate(in_file):
                # Omit comment lines or review titles
                if len(line) < 2 or line.startswith(omit_symbols):
                    continue

                # A review has one or multiple <opinion>[(+/-)<strength>]
                # followed by ##<review>. Find beginning of review
                beginning = line.find('##')
                opinion = line[:beginning]
                review = line[beginning + 2:]

                # Extract opinion strengths
                values = re.findall(pattern, opinion)

                # Some (very few, ~ 3) reviews have annotation mistakes
                if len(values) == 0:
                    continue

                # Sum all opinion strengths and binarize result
                net_value = sum(map(lambda x: int(x[1:-1]), values))
                sentiment = 1 if net_value >= 0 else 0

                data += review
                labels += str(sentiment) + '\n'

                if i == 10:
                    break

    data_file = os.path.join(clean_data_path, 'data.txt')
    with open(data_file, 'w') as out_file:
        out_file.write(data)
    print('Saved {}'.format(data_file))

    labels_file = os.path.join(clean_data_path, 'labels.txt')
    with open(labels_file, 'w') as out_file:
        out_file.write(labels)
    print('Saved {}'.format(labels_file))

    return data_file, labels_file

def load_embeddings():
    """Load GloVe embeddings and export numpy array and word index (as dict)"""
    embeddings_file_base = 'glove.6B.50d'
    embeddings_file = embeddings_file_base + '.txt'

    # Initialize vocabulary with special tokens
    word_to_id = defaultdict(lambda: len(word_to_id))
    _ = word_to_id[PAD]
    _ = word_to_id[UNK]

    embeddings_list = []

    print('Reading embeddings...')
    with open(os.path.join(embeddings_path, embeddings_file)) as emb_file:
        for i, line in enumerate(emb_file):
            # First element is word, the rest is embedding as sequence of float
            values = line.strip().split()
            _ = word_to_id[values[0]]
            embeddings_list.append(list(map(float, values[1:])))

    # Store in numpy array and serialize
    embeddings = np.zeros([len(word_to_id), len(embeddings_list[0])])
    embeddings[2:] = embeddings_list
    np.save(os.path.join(embeddings_path, embeddings_file_base), embeddings)
    print('Saved {}'.format(embeddings_file_base + '.npy'))

    # Serialize dictionary
    index_name = embeddings_file_base + '.index'
    index_file = os.path.join(embeddings_path, index_name)
    pickle.dump(dict(word_to_id), open(index_file, 'wb'))
    print('Saved {}'.format(index_name))

    return index_file

def serialize_data(data_file, labels_file, index_file):
    """Read clean data and labels file and export TFRecords containing
    sentences as sequences of word IDs, review label and length.
    The data is shuffled and split into training and test sets. Each set is
    exported as a TFRecord.
    Args:
        data_file: str, path to the clean data file
        labels_file: str, path to the labels file
        index_file: str, path to the serialized index written by
            load_embeddings()
    """
    word_to_id = pickle.load(open(index_file, 'rb'))
    data = []
    with open(data_file) as file:
        for line in file:
            words = line.split()
            int_words = [word_to_id.get(w, word_to_id[UNK]) for w in words]
            data.append(int_words)

    with open(labels_file) as file:
        labels = [int(line) for line in file]

    # Shuffle samples
    np.random.seed(42)
    idx = np.random.choice(np.arange(start=0, stop=len(data)),
                                     size=len(data), replace=True)
    data = [data[i] for i in idx]
    labels = [labels[i] for i in idx]

    # Split into train and test
    train_split = 0.8
    train_end_idx = int(train_split * len(data))

    # Serialize splits
    save_to_tfrecord(data[:train_end_idx], labels[:train_end_idx], 'train')
    save_to_tfrecord(data[train_end_idx:], labels[train_end_idx:], 'test')

def parse_example(example_proto):
    """Parse a serialized SequenceExample read from a TFRecord.
    Args:
        example_proto: str, a single binary serialized SequenceExample proto.
    Returns:
        length: 1-D Tensor of length 1
        label: 1-D Tensor of length 1
        words: 1-D Tensor of variable length
    """
    # Context: sequence length and label
    # We parse these as rank 1 tensors (instead of rank 0)
    # so they can be batched
    context_features = {
        LEN_FEAT_NAME: tf.FixedLenFeature([1], dtype=tf.int64),
        LABEL_FEAT_NAME: tf.FixedLenFeature([1], dtype=tf.int64)
    }

    # Feature lists: words
    sequence_features = {
        WORDS_FEAT_NAME: tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context, sequence = tf.parse_single_sequence_example(
        example_proto,
        context_features,
        sequence_features
    )

    length = context[LEN_FEAT_NAME]
    label = context[LABEL_FEAT_NAME]
    words = sequence[WORDS_FEAT_NAME]

    return length, label, words

def make_dataset(tfrecord_file, batch_size):
    """Create a batched Dataset from a TFRecord file
    Args:
        tfrecord_file: str, path to the TFRecord file
        batch_size: int, the size of each mini-batch
    Returns:
        A tf.data.TFRecordDataset
    """
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(parse_example)
    padded_shapes = (
        tf.TensorShape([1]),  # length
        tf.TensorShape([1]),  # label
        tf.TensorShape([None])  # words
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes)

    return dataset

if __name__ == '__main__':
    index_file = load_embeddings()
    data_file, labels_file = clean_data()
    serialize_data(data_file, labels_file, index_file)
