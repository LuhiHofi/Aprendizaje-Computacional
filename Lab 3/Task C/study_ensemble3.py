from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gwo import gwo
from performance import printPerformance

import random

#Load data
X_train = np.load("./data/X_train.npy")
y_train = np.load("./data/y_train.npy")
X_val = np.load("./data/X_val.npy")
y_val = np.load("./data/y_val.npy")
X_test= np.load("./data/X_test.npy")
y_test = np.load("./data/y_test.npy")

variable_translation = {
    "Age.1": 0,
    "Temp": 1,
    "OsSats": 2,
    "Lympho": 3,
    "WBC": 4,
    "Plts": 5,
    "Creatinine": 6,
    "MAP": 7,
    "Sodium": 8,
    "ALT": 9,
    "AST": 10,
    "INR": 11,
    "BUN": 12,
    "Troponin": 13,
    "CrctProtein": 14,
    "Ddimer": 15,
    "Glucose": 16,
    "Ferritin": 17,
    "Procalcitonin": 18,
    "IL6": 19
}

selected_attributes2 = ['MAP', 'OsSats', 'Age.1', 'Procalcitonin', 'Ddimer']

selected_attributes2 = [variable_translation[var] for var in selected_attributes2]

def objective_ensemble_reduced2(trial):
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
    
    prob1 = clf1.fit(X_train[selected_attributes2], y_train).predict_proba(X_val[selected_attributes2])[::,1]
    prob2 = clf2.fit(X_train[selected_attributes2], y_train).predict_proba(X_val[selected_attributes2])[::,1]
    prob3 = clf3.fit(X_train[selected_attributes2], y_train).predict_proba(X_val[selected_attributes2])[::,1]
    ensemble_prob = prob1*norm_w1 + prob2*norm_w2 + prob3*norm_w3
    aucroc = roc_auc_score(y_val, ensemble_prob)
    return aucroc

#best_position = gwo(objective_ensemble_reduced2, max_iter=1000, n=50, dim=1, minx=0.1, max=0.9)

#Results
study_ensemble_reduced2 = {'weight1': 0.7661967917261097,
                           'weight2': 0.1629127976463277,
                           'weight3': 0.8959127458815679}

tuned_norm_w1_r2 = study_ensemble_reduced2['weight1']
tuned_norm_w2_r2 = study_ensemble_reduced2['weight2']
tuned_norm_w3_r2 = study_ensemble_reduced2['weight3']


clf1_reduced2 = GradientBoostingClassifier(random_state=0, 
                                  learning_rate=0.05,
                                  max_depth=3,
                                  min_samples_leaf=2,
                                  n_estimators=290, 
                                  subsample=0.66)

clf2_reduced2 = RandomForestClassifier(n_jobs=-1,
                              max_depth=4,
                              max_features=0.75, 
                              min_samples_leaf=2,
                              n_estimators=108)
    
clf3_reduced2 = ExtraTreesClassifier(n_jobs=-1, 
                            max_depth=4,
                            max_features=0.85,
                            min_samples_leaf=1,
                            n_estimators=294)

print("Training the ensemble model...")

prob1_reduced2 = clf1_reduced2.fit(X_train[:, selected_attributes2], y_train).predict_proba(X_test[:,selected_attributes2])[::,1]
prob2_reduced2 = clf2_reduced2.fit(X_train[:, selected_attributes2], y_train).predict_proba(X_test[:,selected_attributes2])[::,1]
prob3_reduced2 = clf3_reduced2.fit(X_train[:, selected_attributes2], y_train).predict_proba(X_test[:,selected_attributes2])[::,1]

prob_ensemble_reduced2 = prob1_reduced2*tuned_norm_w1_r2 + prob2_reduced2*tuned_norm_w2_r2 + prob3_reduced2*tuned_norm_w3_r2

print("Performance of the ensemble model...")

ensemble_aucroc_reduced2 = roc_auc_score(y_test, prob_ensemble_reduced2)
clf1_aucroc_reduced2 =  roc_auc_score(y_test, prob1_reduced2)
clf2_aucroc_reduced2 =  roc_auc_score(y_test, prob2_reduced2)
clf3_aucroc_reduced2 =  roc_auc_score(y_test, prob3_reduced2)

print(f"ensemble: {ensemble_aucroc_reduced2}")
print(f"clf1: {clf1_aucroc_reduced2}")
print(f"clf2: {clf2_aucroc_reduced2}")
print(f"clf3: {clf3_aucroc_reduced2}")
# print(f"clf4: {clf4_aucroc}")

print("Feature importance of the ensemble model...")

df_en_r2 = printPerformance(y_test, prob_ensemble_reduced2, auc_only=False)
df1_r2 = printPerformance(y_test, prob1_reduced2, auc_only=False)
df2_r2 = printPerformance(y_test, prob2_reduced2, auc_only=False)
df3_r2 = printPerformance(y_test, prob3_reduced2, auc_only=False)
df_r2 = pd.DataFrame([df_en_r2, df1_r2, df2_r2, df3_r2])
df_r2.columns = ['AUCROC', 'AUCPR', 'ACC', 'BA', 'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK']
df_r2.to_csv("./outcomes/performance_r2.csv", index=None)