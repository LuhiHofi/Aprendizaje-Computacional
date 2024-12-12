from gwo import gwo
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

import random


X_train = np.load("./data/X_train.npy")
y_train = np.load("./data/y_train.npy")
X_val = np.load("./data/X_val.npy")
y_val = np.load("./data/y_val.npy")

def objective_ensemble(trial):
    clf1 = GradientBoostingClassifier(random_state=0, 
                                  learning_rate=0.05,
                                  max_depth=3,
                                  min_samples_leaf=2,
                                  n_estimators=290, 
                                  subsample=0.66)
    clf2 = RandomForestClassifier(n_jobs=-1,
                                  max_depth=4,
                                  max_features=0.75, 
                                  min_samples_leaf=2,
                                  n_estimators=108)
    
    clf3 = ExtraTreesClassifier(n_jobs=-1, 
                                max_depth=4,
                                max_features=0.85,
                                min_samples_leaf=1,
                                n_estimators=294)

    # w1 = trial.suggest_float('weight1', 0.1, 0.9)
    # w2 = trial.suggest_float('weight2', 0.1, 0.9)
    # w3 = trial.suggest_float('weight3', 0.1, 0.9)
    w1 = random.uniform(0.1, 0.9)
    w2 = random.uniform(0.1, 0.9)
    w3 = random.uniform(0.1, 0.9)
    norm_w1, norm_w2, norm_w3 = w1/(w1+w2+w3), w2/(w1+w2+w3), w3/(w1+w2+w3)
    
    prob1 = clf1.fit(X_train, y_train).predict_proba(X_val)[::,1]
    prob2 = clf2.fit(X_train, y_train).predict_proba(X_val)[::,1]
    prob3 = clf3.fit(X_train, y_train).predict_proba(X_val)[::,1]
    ensemble_prob = prob1*norm_w1 + prob2*norm_w2 + prob3*norm_w3
    aucroc = roc_auc_score(y_val, ensemble_prob)
    return aucroc

best_position = gwo(objective_ensemble, max_iter=1000, n=50, dim=1, minx=0.1, maxx=0.9)
