#!/usr/bin/env python3

import os
import random
import sys

import dill as pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

import tabio.column_detection
import tabio.config
import tabio.data_loader
import tabio.frontend
import tabio.pascalvoc
import tabio.split_lines


def load():
    return pickle.load(open(os.path.join(os.path.dirname(__file__), "lexical_model.pickle"), "rb"))


def create_lexical_features(lexical_model, lines):
    [tfidf, ts, tfidfw, tsw] = lexical_model
    line_text = [li.text for li in lines]
    text_feat = np.hstack((ts.transform(tfidf.transform(
        line_text)), tsw.transform(tfidfw.transform(line_text))))
    return text_feat


def get_classification(text_list, l_model):
    [tfidf, ts, reg] = l_model
    result = reg.predict(ts.transform(tfidf.transform(text_list)))
    return result


def create_training_text(page):
    labeled_boxes = tabio.pascalvoc.read(page.label_fname)

    lines = tabio.frontend.read_lines(page)
    columns = map(lambda l: tabio.column_detection.fake_column_detection(
        l, labeled_boxes), lines)
    lines = tabio.split_lines.split_lines(lines, columns)

    lines = list(filter(lambda l: l is not None, lines))
    labels = map(lambda l: tabio.column_detection.read_line_classification(
        l, labeled_boxes), lines)
    labels = list(filter(lambda l: l is not None, labels))
    labels = list(map(lambda x: tabio.config.interpret_label(x)[1], labels))
    if len(labels) == 0 or len(lines) == 0:
        return ([], [])
    zipped = filter(lambda x: x[1] is not None, zip(lines, labels))
    [lines, labels] = list(zip(*list(zipped)))
    return (list(lines), list(labels))


if __name__ == '__main__':

    print('loading...')
    lines = []
    labels = []
    for page in list(tabio.data_loader.training_pages()):
        print(page)
        li, la = create_training_text(page)
        lines.extend(li)
        labels.extend(la)
    indices = [i for i in list(range(len(labels)))
               if labels[i] in tabio.config.class_map.keys()]
    lines = [lines[i][0] for i in indices]
    labels = [labels[i] for i in indices]
    lines_train = lines[:int(0.9*len(indices))]
    labels_train = labels[:int(0.9*len(indices))]
    lines_test = lines[int(0.9*len(indices)):]
    labels_test = labels[int(0.9*len(indices)):]

    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(
        1, 2), strip_accents="unicode", decode_error="ignore")
    X = tfidf.fit_transform(lines_train)
    ts = TruncatedSVD(n_components=100)
    ts.fit(X)
    tfidfw = TfidfVectorizer(analyzer="char", ngram_range=(
        1, 2), strip_accents="unicode", decode_error="ignore")
    Xw = tfidfw.fit_transform(lines_train)
    tsw = TruncatedSVD(n_components=100)
    tsw.fit(Xw)
    X = np.hstack((ts.transform(X), tsw.transform(Xw)))
    cfr = RFC(n_jobs=20)
    cfr.fit(X, labels_train)
    X_test = ts.transform(tfidf.transform(lines_test))
    X_testw = tsw.transform(tfidfw.transform(lines_test))
    X_test = np.hstack((X_test, X_testw))
    print(cfr.score(X_test, labels_test))
    Y_test = cfr.predict(X_test)
    print(classification_report(labels_test, Y_test))
    pickle.dump([tfidf, ts, tfidfw, tsw], open("lexical_model.pickle", "wb"))
