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

data_path = './data'
embeddings_path = './embeddings'
raw_data_path = os.path.join(data_path, 'raw')
clean_data_path = os.path.join(data_path, 'clean')

PAD = '<PAD>'
UNK = '<UNK>'

def remove_dir(dir):
    if os.path.exists(dir):
        rmtree(dir)
    os.mkdir(dir)

def clean_data():
    files = listdir(raw_data_path)
    files.remove('Readme.txt')

    # Remove any exported clean files if they exist
    remove_dir(clean_data_path)

    omit_symbols = ('*', '[t]', '##')

    # To detect one or more annotations like [-1]
    pattern = re.compile('\[[+-]{1}[0-9]\]')
    data = ''
    labels = ''

    for file in files:
        with open(os.path.join(raw_data_path, file)) as in_file:
            for line in in_file:
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

    data_file = os.path.join(clean_data_path, 'data.txt')
    with open(data_file, 'w') as out_file:
        out_file.write(data)
    print('Saved {}'.format(data_file))

    labels_file = os.path.join(clean_data_path, 'labels.txt')
    with open(labels_file, 'w') as out_file:
        out_file.write(labels)
    print('Saved {}'.format(labels_file))

def load_embeddings():
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
    vocab_filename = embeddings_file_base + '.index'
    pickle.dump(dict(word_to_id),
                open(os.path.join(embeddings_path, vocab_filename), 'wb'))
    print('Saved {}'.format(vocab_filename))

if __name__ == '__main__':
    clean_data()
    load_embeddings()
