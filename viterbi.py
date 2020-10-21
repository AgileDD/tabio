#!/usr/bin/env python3

# This implements viterbi search to improve the results from the line classifier
# The output of this search will be a classification of each line on a page
# Improvements come from the additional knowledge of line orderings from the line_trigram model
#
# A table detector can then use the classifications to lines representing tables on the page

import data_loader
import line_trigram
import frontend
import pascalvoc
import column_detection
import torch
import numpy as np
from PIL import Image
from collections import namedtuple
import os.path

import cProfile

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw
import csv_file
import config
import line_classifier
import split_lines
import itertools
import align
import scipy.signal


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
 
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] + emit_p(y, obs[0])
        path[y] = [y]
    
    # alternative Python 2.7+ initialization syntax
    # V = [{y:(start_p[y] * emit_p[y][obs[0]]) for y in states}]
    # path = {y:[y] for y in states}
 
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
 
        for y in states:
            (prob, state) = max((V[t-1][y0] + trans_p(y0, y) + emit_p(y, obs[t]), y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
            
 
        # Don't need to remember the old paths
        path = newpath
        #print('.', end='', flush=True)
        
    #print('')
    #print_dptable(V)
    (prob, state) = max((V[t][y], y) for y in states)
    return (prob, path[state])

def print_dptable(V):
    s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
    for y in V[0]:
        s += "%.5s: " % y
        s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
        s += "\n"
    print(s)

classes = config.classes

# runs viterbi search on a page and returns a list of classes, one per line
def search_page(transition_model, emission_model, features):
    State = namedtuple('State', ['class_id', 'context_id'])

    class_ids = list(range(len(classes)))

    states = []

    #create initial states
    for i in class_ids:
        states.append(State(i, -1))
    
    #create trigram states
    for i in class_ids:
        for j in class_ids:
            states.append(State(i, j))

    state_ids = list(range(len(states)))

    log_zero = float('-inf')
    lm_weight = .05

    transition_probabilities = []

    def transition_probability(prev_id, next_id):
        prev = states[prev_id]
        next = states[next_id]
        #never move back to an initial state
        if next.context_id == -1:
            return log_zero

        #don't allow moving to a state where the context doesn't match
        if prev.class_id != next.context_id:
            return log_zero

        word = classes[next.class_id]
        context1_id = prev.context_id
        if context1_id == -1:
            context = (classes[prev.class_id],)
        else:
            context = (classes[context1_id], classes[prev.class_id])
        try:
            return lm_weight * transition_model.logscore(word, context)
        except:
            return log_zero

    for i in state_ids:
        probs = []
        for j in state_ids:
            probs.append(transition_probability(i, j))
        transition_probabilities.append(probs)

    def trans_p(prev_id, next_id):
        t = transition_probabilities[prev_id][next_id]
        return t

    emission_scores = []
    features = list(features)

    best_classes = []

    emission_scores = line_classifier.eval(emission_model, features)
    emission_scores = list(map(lambda s: s.detach().cpu().numpy(), emission_scores))

    def emit_p(state_id, feature_id):
        state = states[state_id]
        e = emission_scores[feature_id][state.class_id]
        return e

    start_probabilities = []
    for i in range(len(classes)):
        start_probabilities.append(lm_weight * transition_model.logscore(classes[i], None))

    start_probabilities += [log_zero] * (len(classes) * len(classes))

    feature_ids = list(range(len(features)))
    prob, path = viterbi(feature_ids, state_ids, start_probabilities, trans_p, emit_p)

    class_ids = [states[p].class_id for p in path]
    orig_ids = class_ids
    class_ids = scipy.signal.medfilt(class_ids, 5)
    class_ids = [int(i) for i in class_ids]

    #for o,f in zip(orig_ids, class_ids):
    #    print(o,f)

    hypothesis = list(map(lambda i: classes[i], class_ids))
    return hypothesis


def page_truth(page):
    labeled_boxes = pascalvoc.read(page.label_fname)
    lines = frontend.read_lines(page)
    columns = [column_detection.fake_column_detection(l, labeled_boxes) for l in lines]
    lines = split_lines.split_lines(lines, columns)

    def GetClass(classification):
            if classification is None:
                return 'unknown'
            return classification.split('-')[1]

    truth = list(map(lambda l: GetClass(column_detection.read_line_classification(l, labeled_boxes)), lines))
    return (lines, truth)


if __name__ == '__main__':

    transition_model = line_trigram.load()
    emission_model = line_classifier.load()
    column_model = column_detection.load()

    print(f"{'status':<10}{'hypothesis':<24}{'reference':<24}")

    for page in data_loader.test_pages():
        features, lines = frontend.create(page, lambda ls, ms: column_detection.eval(column_model, ms))

        hypothesis = search_page(transition_model, emission_model, features)

        reference_lines, truth = page_truth(page)

        aligned_ref, aligned_hyp, alignment_status = align.align(truth, hypothesis)
        for r,h,s in zip(aligned_ref, aligned_hyp, alignment_status):
            if r is None:
                r = '-'
            if h is None:
                h =  '-'
            print(f'{s!s:<10}{h!s:<24}{r!s:<24}')
        
