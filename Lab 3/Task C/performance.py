import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


#=============================================
def printPerformance(labels, probs, threshold:float=None, decimal:int=None, printout=False, auc_only=True):
    if threshold != None:
        assert threshold < 0 and threshold > 1, "threshold must be in the range [0 to 1]"
        predicted_labels = np.array([1 if prob >= threshold else 0 for prob in probs]).astype(int)
    else:
        # if predicted label says > 1, then it is 1, otherwise 0
        predicted_labels = np.array([1 if prob >= 0.5 else 0 for prob in probs]).astype(int)

        labels = labels.astype(int)
    #------------------------------------------------    
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    if decimal != None:
        assert decimal <= 8, "decimal must be int and in the range [0 to 8]"
        d = decimal
    else:
        d = 4
    #------------------------------------------------
    aucroc = round(roc_auc_score(labels, probs), d)
    aucpr  = round(average_precision_score(labels, probs), d)
    if auc_only:
        if printout:
            print('AUCROC: {}'.format(aucroc))
            print('AUCPR: {}'.format(aucpr))
        return aucroc, aucpr
    else:
        acc    = round(accuracy_score(labels, predicted_labels), d) 
        ba     = round(balanced_accuracy_score(labels, predicted_labels), d)
        mcc    = round(matthews_corrcoef(labels, predicted_labels), d)
        sen    = round(tp / (tp + fn), d)
        spe    = round(tn / (tn + fp), d)
    #     pre    = round(tp / (tp + fp), d)
        pre    = round(precision_score(labels, predicted_labels), d)
    #     f1     = round(2*pre*sen / (pre + sen), d)
        f1     = round(f1_score(labels, predicted_labels), d)
        ck     = round(cohen_kappa_score(labels, predicted_labels), d)
    #------------------------------------------------
        if printout:
            print('AUCROC: {}'.format(aucroc))
            print('AUCPR: {}'.format(aucpr))
            print('ACC: {}'.format(acc))
            print('BA: {}'.format(ba))
            print('SN/RE : {}'.format(sen))
            print('SP: {}'.format(spe))
            print('PR: {}'.format(pre))
            print('MCC: {}'.format(mcc))
            print('F1: {}'.format(f1))
            print('CK: {}'.format(ck))
        #------------------------------------------------        
        return aucroc, aucpr, acc, ba, sen, spe, pre, mcc, f1, ck
        
        
        