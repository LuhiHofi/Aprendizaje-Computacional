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


X_train = pd.DataFrame(X_train)

#Objective function
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

#best_position = gwo(objective_ensemble, max_iter=1000, n=50, dim=1, minx=0.1, maxx=0.9)
#print("Best position: ", best_position)

#Results
study_ensemble = {'weight1': 0.8714467832442298,
                  'weight2': 0.10055484383424314,
                  'weight3': 0.4926102019970062}

tuned_norm_w1 = study_ensemble['weight1']
tuned_norm_w2 = study_ensemble['weight2']
tuned_norm_w3 = study_ensemble['weight3']


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

print("Training the ensemble model...")

prob1 = clf1.fit(X_train, y_train).predict_proba(X_test)[::,1]
prob2 = clf2.fit(X_train, y_train).predict_proba(X_test)[::,1]
prob3 = clf3.fit(X_train, y_train).predict_proba(X_test)[::,1]

prob_ensemble = prob1*tuned_norm_w1 + prob2*tuned_norm_w2 + prob3*tuned_norm_w3 

print("Performance of the ensemble model...")

ensemble_aucroc = roc_auc_score(y_test, prob_ensemble)
clf1_aucroc =  roc_auc_score(y_test, prob1)
clf2_aucroc =  roc_auc_score(y_test, prob2)
clf3_aucroc =  roc_auc_score(y_test, prob3)

df_en = printPerformance(y_test, prob_ensemble, auc_only=False)
df1 = printPerformance(y_test, prob1, auc_only=False)
df2 = printPerformance(y_test, prob2, auc_only=False)
df3 = printPerformance(y_test, prob3, auc_only=False)
df = pd.DataFrame([df_en, df1, df2, df3])
df.columns = ['AUCROC', 'AUCPR', 'ACC', 'BA', 'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK']
df.to_csv("./outcomes/performance_full.csv", index=None)


feature_importances_gb  = clf1.feature_importances_
feature_importances_rf  = clf2.feature_importances_
feature_importances_ert = clf3.feature_importances_
attributes = list(variable_translation.keys())

print("Feature importance of the ensemble model...")

fea_im_pair_gb = sorted(zip(feature_importances_gb, attributes), reverse=True)
fea_im_gb_fullset_values = [fea_im_pair_gb[i][0] for i in range(len(fea_im_pair_gb))]
fea_im_gb_fullset_columns = [fea_im_pair_gb[i][1] for i in range(len(fea_im_pair_gb))]

fea_im_pair_rf = sorted(zip(feature_importances_rf, attributes), reverse=True)
fea_im_rf_fullset_values = [fea_im_pair_rf[i][0] for i in range(len(fea_im_pair_rf))]
fea_im_rf_fullset_columns = [fea_im_pair_rf[i][1] for i in range(len(fea_im_pair_rf))]

fea_im_pair_ert = sorted(zip(feature_importances_ert, attributes), reverse=True)
fea_im_ert_fullset_values = [fea_im_pair_ert[i][0] for i in range(len(fea_im_pair_ert))]
fea_im_ert_fullset_columns = [fea_im_pair_ert[i][1] for i in range(len(fea_im_pair_ert))]

fig, axs = plt.subplots(3, figsize=(15,10), tight_layout=True)

fea_im_fullset_columns = [fea_im_gb_fullset_columns, fea_im_rf_fullset_columns, fea_im_ert_fullset_columns]
fea_im_fullset_values = [fea_im_gb_fullset_values, fea_im_rf_fullset_values, fea_im_ert_fullset_values]

colors=['C0','C1','C2']
# creating the bar plot
print("Creating the feature importance plot...")
for i in range(3):
    axs[i].bar(fea_im_fullset_columns[i], fea_im_fullset_values[i], width = 0.8, color=colors[i])

for ax in axs.flat:
    ax.tick_params(axis='x', labelrotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set(xlabel='Features (Variables)', ylabel='Feature Importance', ylim =(0, 0.5))
    ax.xaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontsize(15)
    

fig.savefig("./outcomes/fea_im_fullset.pdf")

gb = sorted(zip(feature_importances_gb, attributes), reverse=True)[:10]
print("Feature importance of the gb...", gb)
rf = sorted(zip(feature_importances_rf, attributes), reverse=True)[:10]
print("Feature importance of the rf...", rf)
ert = sorted(zip(feature_importances_ert, attributes), reverse=True)[:10]
print("Feature importance of the ert...", ert)