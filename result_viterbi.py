import string
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import config
import sys

def transform_classes(sequence):
    result = []
    for i in sequence:
        if i == '-':
            result.append(i)
        elif 'Tab' in i or 'Frame' in i:
            result.append('Table')
        else:
            result.append('NotTable')
    return result

def pr_rec(status, hypothesis, reference):
    reference = transform_classes(reference)
    hypothesis = transform_classes(hypothesis)

    total_reference = 0
    total_hypothesis = 0
    correct = 0
    substitutions = 0
    deletions = 0
    insertions = 0
    


    for s, h, r in zip(status, hypothesis, reference):
        # dont care about inserting non table line
        # dont care about deleting non table line
        if s == 'i':# and h == 'Table':
            insertions += 1
            total_hypothesis += 1
        if s == 'd':# and r == 'Table':
            deletions += 1
            total_reference += 1
        if s == 'c':
            correct += 1
            total_reference += 1
            total_hypothesis += 1
        if s == 's':
            #substitution status was calculated before we mapped all classes to 'Table' or 'NoneTable'
            # we need to recheck if it is actually a substitution
            if r == h:
                correct += 1
            else:
                substitutions += 1
            total_reference += 1
            total_hypothesis += 1

    precision = float(correct) / float(correct + substitutions + insertions)
    recall = float(correct) / float(correct + substitutions + deletions)

    f1 = 2.0 * precision * recall / (precision + recall)

    print(f"Precision = {precision}")
    print(f"Recall    = {recall}")
    print(f"F1-score  = {f1}")
    tp=0
    fn=0
    fp=0
    allref = reference
    allhyp = hypothesis
    for i in range(len(allref)):
        if allref[i]=="Table":
            if allhyp[i]=="Table":
                tp = tp + 1.0
            else:
                fn = fn + 1.0
        else:
            if allhyp[i]=="Table":
                fp = fp + 1.0
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    print("Precision="+str(p))
    print("Recall="+str(r))
    print("F1-Score="+str(2.0*p*r/(p+r)))


def confusionMatrix(allref,allvi):
    outputs = ''
    classes = config.classes
    array_vi = numpy.zeros(shape=(len(classes),len(classes))) 
    dicClasses = {classe: i for i, classe in enumerate(classes)}
    for ref,vi in zip(allref, allvi):
        if(ref not in classes or vi not in classes):
            continue
        array_vi[dicClasses[str(vi)]][dicClasses[str(ref)]] += 1
        #array[line][column] : array[y][x]

    df_cm_vi = pd.DataFrame(array_vi, classes, classes)


    plt.figure(figsize=(90,95))
    params = {'axes.labelsize': 50, 'xtick.labelsize' : 50 , 'ytick.labelsize' : 50, 'legend.fontsize' : 50}
    plt.rcParams.update(params)
    sn.heatmap(df_cm_vi, annot=True, annot_kws={"size": 50}) # font size
    plt.title('After viterbi',fontsize = 120)
    plt.xlabel('References')
    plt.ylabel('Tests')
    plt.savefig(outputs+'After_Viterbi')


    #plt.show()

outputs = ''
in_fname = 'vlog.txt'
if len(sys.argv) > 1:
    in_fname = sys.argv[1]

lines = open(in_fname).readlines()
lines = [x.strip() for x in lines]
allwords = [x.split() for x in lines]
allwords = [x for x in allwords if len(x) == 3]
#print(allwords)
[status, hypothesis, reference] = zip(*allwords)

pr_rec(status, hypothesis, reference)

#confusionMatrix(ref,vi)
