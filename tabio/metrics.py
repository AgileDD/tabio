import numpy as np
from sklearn.metrics import roc_curve as rocc


def roc_curve(true_labels,predict_proba,plus_classes = [0]):
       n_classes = predict_proba.shape[0]
       neg_classes = set(range(n_classes))-set(plus_classes)
       print("probab_shape = "+str(predict_proba.shape))
       plus_scores = np.sum(predict_proba[:,plus_classes],axis=1)
       print("plus_scores = "+str(plus_scores.shape))
       plus_labels = [1 if l in plus_classes else 0 for l in true_labels]
       print(plus_labels)
       fpr,tpr,_ = rocc(plus_labels,plus_scores)
       return fpr,tpr
       
       
