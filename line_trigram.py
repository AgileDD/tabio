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

# creates a list containing line classes
#
# The listused to train a language model to predict 
# the order of classes
def create_training_text(lines, labeled_boxes):
    current_left = []
    current_right = []
    status = 'double'

    out_classes = []
    (width, _) = csv_file.size(lines)

    for line in lines:
        line_category = split_lines.classify_line(line, labeled_boxes)
        if line_category is None:
            continue

        column_type, label = line_category.split('-')
        if column_type == 'SingleColumn':
            if status == 'double':
                out_classes += current_left
                out_classes += current_right
                current_left = []
                current_right = []
                status = 'single'
            out_classes.append(label)
        else:
            status = 'double'
            #handle double column
            l,r = split_lines.split_line(line, width/2.0)
            if l is not None:
                l_category = split_lines.classify_line(l, labeled_boxes)
                if l_category is not None:
                    _, l_label = l_category.split('-')
                    current_left.append(l_label)

            if r is not None:
                r_category = split_lines.classify_line(r, labeled_boxes)
                if r_category is not None:
                    _, r_label = r_category.split('-')
                    current_right.append(r_label)

    out_classes += current_left
    out_classes += current_right
    return out_classes

def load():
    with open('line_ngram.pkl', 'rb') as fin:
        return pickle.load(fin)


if __name__ == '__main__':
    training_text = []
    test_text = []

    for page in data_loader.all_pages():
        lines = csv_file.read_csv(page.csv_fname)
        lines = csv_file.remove_margin(csv_file.group_lines_spacially(lines))
        labeled_boxes = pascalvoc.read(page.label_fname)

        page_classes = create_training_text(lines, labeled_boxes)
        if page.hash in data_loader.test_hashes:
            test_text.append(page_classes)
        else:
            training_text.append(page_classes)

    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, training_text)
    model = KneserNeyInterpolated(n)
    model.fit(train_data, padded_sents)

    with open('line_ngram1.pkl', 'wb') as fout:
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