#!/usr/bin/env python3

import data_loader
import line_trigram
import frontend
import torch
import numpy as np
import model as modl
from PIL import Image



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
    for p in pages:
        print(p.hash, p.page_number)


    feature_vectors = list(map(frontend.create, pages))

    print(len(pages))
    print(len(feature_vectors))

    def trans_p(prev, next):
        # todo: figure out how to incorporate trigram scoring
        return .01 * transition_model.logscore(classes[next], (classes[prev],))



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


        def emit_p(state, feature):
            return emission_scores[features.index(feature)][0][state]

        prob, path = viterbi(features, list(range(len(classes))), [1.0]*len(classes), trans_p, emit_p)
        print(page.hash, page.page_number)
        print(list(map(lambda p: classes[p], path)))
