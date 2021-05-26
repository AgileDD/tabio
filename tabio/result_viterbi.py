import string
import sys

import matplotlib.pyplot as plt
import numpy
import pandas as pd
import seaborn as sn

import tabio.config

def transform_classes(sequence):
    result = []
    for i in sequence:
        if i == '-':
            result.append(i)
        # elif i=="Table" or 'TableSparseMulticolumn' in i or "TableSparseColumnHeader" in i or "TableSparse" in i:
        elif "Table" in i:
        # elif i=="Table" or "Frame" in i:
            result.append('Table')
        else:
            result.append('Else')
        print(i+" "+result[-1])
    return result

def pr_rec(status, hypothesis, reference):
    print(reference[:100])
    print(hypothesis[:100])
    reference = transform_classes(reference)
    hypothesis = transform_classes(hypothesis)

    tp=0
    fn=0
    fp=0
    tn = 0
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
            else:
                tn = tn + 1.0
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    print("Precision="+str(p))
    print("Recall="+str(r))
    print("F1-Score="+str(2.0*p*r/(p+r)))
    print("False positive rate="+str(fp/(tn+fp)))


def confusionMatrix(allref,allvi):
    outputs = ''
    classes = tabio.config.classes
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
allwords = [[x.split()[0],x.split()[1],"".join(x.split()[2:])] for x in lines]
allwords = [x for x in allwords if len(x) == 3]
#print(allwords)
[status, hypothesis, reference] = zip(*allwords)

pr_rec(status, hypothesis, reference)

#confusionMatrix(ref,vi)
