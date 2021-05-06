#!/usr/bin/env python3

import sys
import os
import split_lines
import csv_file
import pascalvoc
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import flatten
from nltk.lm import KneserNeyInterpolated
from statistics import mean
import dill as pickle
import data_loader
import frontend
import column_detection
import config

# creates a list containing line classes
#
# The listused to train a language model to predict 
# the order of classes
def create_training_text(page):
    labeled_boxes = pascalvoc.read(page.label_fname)

    lines = frontend.read_lines(page)
    columns = map(lambda l: column_detection.fake_column_detection(l, labeled_boxes), lines)
    lines = split_lines.split_lines(lines, columns)

    lines = filter(lambda l: l is not None, lines)
    labels = map(lambda l: column_detection.read_line_classification(l, labeled_boxes), lines)
    labels = filter(lambda l: l is not None, labels)
    return map(lambda l: config.class_map[config.interpret_label(l)[1]], labels)

def load():
    with open(os.path.join(os.path.dirname(__file__), 'line_ngram.pkl'), 'rb') as fin:
        return pickle.load(fin)


if __name__ == '__main__':
    training_text = []
    test_text = []

    print('loading...')
    for page in data_loader.training_pages():

        page_classes = create_training_text(page)
        if page.hash in data_loader.test_hashes:
            test_text.append(list(page_classes))
        else:
            training_text.append(list(page_classes))
    print(training_text[:20])
    print('training...')
    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, training_text)
    model = KneserNeyInterpolated(n)
    model.fit(train_data, padded_sents)

    with open('line_ngram.pkl', 'wb') as fout:
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
