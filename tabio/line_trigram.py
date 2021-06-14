#!/usr/bin/env python3

import os
import sys
from statistics import mean

import dill as pickle
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import flatten

import tabio.column_detection
import tabio.config
import tabio.csv_file
import tabio.data_loader
import tabio.frontend
import tabio.pascalvoc
import tabio.split_lines

# creates a list containing line classes
#
# The listused to train a language model to predict
# the order of classes


def create_training_text(page):
    labeled_boxes = tabio.pascalvoc.read(page.label_fname)

    lines = tabio.frontend.read_lines(page)
    columns = map(lambda l: tabio.column_detection.fake_column_detection(
        l, labeled_boxes), lines)
    lines = tabio.split_lines.split_lines(lines, columns)

    lines = filter(lambda l: l is not None, lines)
    labels = map(lambda l: tabio.column_detection.read_line_classification(
        l, labeled_boxes), lines)
    labels = filter(lambda l: l is not None, labels)
    return map(lambda l: tabio.config.class_map[tabio.config.interpret_label(l)[1]], labels)


def load():
    with open(os.path.join(os.path.dirname(__file__), 'models', 'line_ngram.pt'), 'rb') as fin:
        return pickle.load(fin)


if __name__ == '__main__':
    training_text = []
    test_text = []

    print('loading...')
    for page in tabio.data_loader.training_pages():

        page_classes = create_training_text(page)
        if page.hash in tabio.data_loader.test_hashes:
            test_text.append(list(page_classes))
        else:
            training_text.append(list(page_classes))
    print(training_text[:20])
    print('training...')
    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, training_text)
    model = KneserNeyInterpolated(n)
    model.fit(train_data, padded_sents)

    with open(os.path.join('models', 'line_ngram.pt'), 'wb') as fout:
        pickle.dump(model, fout)

    print('generated data: '+' '.join(model.generate(20, random_seed=7)))

    test_data, _ = padded_everygram_pipeline(n, test_text)

    perplexities = []
    for test in test_data:
        try:
            perplexities.append(model.perplexity(test))
        except ZeroDivisionError:
            pass

    print(perplexities)
    print(f'perplexity: {mean(perplexities)}')
