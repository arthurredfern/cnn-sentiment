"""
A script to read the Customer Review dataset and export cleaned files
with binary labels for sentiment classification.
"""

import os
import re
from os import listdir
from shutil import rmtree

def main():
    data_path = './data'
    raw_data_path = os.path.join(data_path, 'raw')
    clean_data_path = os.path.join(data_path, 'clean')
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

    labels_file = os.path.join(clean_data_path, 'labels.txt')
    with open(labels_file, 'w') as out_file:
        out_file.write(labels)

if __name__ == '__main__':
    main()