#!/usr/bin/env python3

import data_loader
import line_trigram
import frontend
import pascalvoc
import column_detection
import torch
import numpy as np
import model as modl
from PIL import Image
from collections import namedtuple



def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
 
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] + emit_p(y, obs[0])
        path[y] = [y]

    print('done initializing ')
    
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
        print('.', end='', flush=True)
        
 
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

if __name__ == '__main__':
    classes = ['Author', 'Equation', 'FigureCaption', 'FigureText', 'Heading', 'PageFooter', 'PageHeader', 'PageNumber', 'Paragraph', 'References', 'Sparse', 'Subtitle', 'TableCaption', 'TableFooter', 'TableSparseColumnHeader', 'TableSparseMulticolumn', 'Title']

    transition_model = line_trigram.load()
    emission_model = torch.load('./trained_net.pth')
    emission_model.eval()
    
    pages = list(data_loader.test_pages())
    feature_vectors = []
    for p in pages:
        print(p.hash, p.page_number)
        labeled_boxes = pascalvoc.read(p.label_fname)
        feature_vectors.append(frontend.create(p, lambda l: column_detection.fake_column_detection(l, labeled_boxes))[0])


    print(len(pages))
    print(len(feature_vectors))

    


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

    log_zero = float('-inf')
    lm_weight = .01

    def trans_p(prev_id, next_id):
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
            return -1.0 * lm_weight * transition_model.logscore(word, context)
        except:
            return log_zero


    for page, features in zip(pages, feature_vectors):
        if not page.page_number == '6':
            continue

        emission_scores = []
        features = list(features)

        for feature in features:
            feature = [np.array(feature, dtype=np.uint8)]
            feature = torch.from_numpy(np.array(feature)/128.0)
            if modl.model_type=="MLP":
                feature = feature.view(feature.shape[0], -1)
            else:
                feature = feature[:,None,:,:]

            emission_scores.append(emission_model(feature).detach().numpy())
        
        print(len(emission_scores))
        print(len(features))


        def emit_p(state_id, feature):
            state = states[state_id]
            return emission_scores[features.index(feature)][0][state.class_id]

        start_probabilities = []
        for i in range(len(classes)):
            start_probabilities.append(lm_weight * transition_model.logscore(classes[i], None))

        start_probabilities += [log_zero] * (len(classes) * len(classes))

        state_ids = list(range(len(states)))
        print(len(start_probabilities))
        print(len(state_ids))
        prob, path = viterbi(features, state_ids, start_probabilities, trans_p, emit_p)
        print(page.hash, page.page_number)
        print(list(map(lambda p: classes[p], path)))
