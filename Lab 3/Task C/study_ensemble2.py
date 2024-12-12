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


selected_attributes1 = ['MAP', 'OsSats', 'Age.1', 'Procalcitonin', 'Troponin', 'IL6', 'Ddimer', 'BUN', 'Temp', 'CrctProtein']

indexAttributes = [variable_translation[var] for var in selected_attributes1]
indexes_to_attributes = {v: k for k, v in variable_translation.items()}

def objective_ensemble_reduced1(trial):
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
    
    prob1 = clf1.fit(X_train[indexAttributes], y_train).predict_proba(X_val[indexAttributes])[::,1]
    prob2 = clf2.fit(X_train[indexAttributes], y_train).predict_proba(X_val[indexAttributes])[::,1]
    prob3 = clf3.fit(X_train[indexAttributes], y_train).predict_proba(X_val[indexAttributes])[::,1]
    ensemble_prob = prob1*norm_w1 + prob2*norm_w2 + prob3*norm_w3
    aucroc = roc_auc_score(y_val, ensemble_prob)
    return aucroc

# best_position = gwo(objective_ensemble_reduced1, max_iter=1000, n=50, dim=1, minx=0.1, maxx=0.9)

#Results
study_ensemble_reduced1 = {'weight1': 0.8727683683283963,
                           'weight2': 0.1001202293990478,
                           'weight3': 0.2919684754184301}

tuned_norm_w1_r1 = study_ensemble_reduced1['weight1']
tuned_norm_w2_r1 = study_ensemble_reduced1['weight2']
tuned_norm_w3_r1 = study_ensemble_reduced1['weight3']


clf1_reduced1 = GradientBoostingClassifier(random_state=0, 
                                  learning_rate=0.05,
                                  max_depth=3,
                                  min_samples_leaf=2,
                                  n_estimators=290, 
                                  subsample=0.66)

clf2_reduced1 = RandomForestClassifier(n_jobs=-1,
                              max_depth=4,
                              max_features=0.75, 
                              min_samples_leaf=2,
                              n_estimators=108)
    
clf3_reduced1 = ExtraTreesClassifier(n_jobs=-1, 
                            max_depth=4,
                            max_features=0.85,
                            min_samples_leaf=1,
                            n_estimators=294)

print("Training the ensemble model...")

prob1_reduced1 = clf1_reduced1.fit(X_train[:, indexAttributes], y_train).predict_proba(X_test[:, indexAttributes])[::,1]
prob2_reduced1 = clf2_reduced1.fit(X_train[:, indexAttributes], y_train).predict_proba(X_test[:, indexAttributes])[::,1]
prob3_reduced1 = clf3_reduced1.fit(X_train[:, indexAttributes], y_train).predict_proba(X_test[:, indexAttributes])[::,1]

prob_ensemble_reduced1 = prob1_reduced1*tuned_norm_w1_r1 + prob2_reduced1*tuned_norm_w2_r1 + prob3_reduced1*tuned_norm_w3_r1

print("Performance of the ensemble model...")
ensemble_aucroc_reduced1 = roc_auc_score(y_test, prob_ensemble_reduced1)
clf1_aucroc_reduced1 =  roc_auc_score(y_test, prob1_reduced1)
clf2_aucroc_reduced1 =  roc_auc_score(y_test, prob2_reduced1)
clf3_aucroc_reduced1 =  roc_auc_score(y_test, prob3_reduced1)

print(f"ensemble: {ensemble_aucroc_reduced1}")
print(f"clf1: {clf1_aucroc_reduced1}")
print(f"clf2: {clf2_aucroc_reduced1}")
print(f"clf3: {clf3_aucroc_reduced1}")
# print(f"clf4: {clf4_aucroc}")

df_en_r1 = printPerformance(y_test, prob_ensemble_reduced1, auc_only=False)
df1_r1 = printPerformance(y_test, prob1_reduced1, auc_only=False)
df2_r1 = printPerformance(y_test, prob2_reduced1, auc_only=False)
df3_r1 = printPerformance(y_test, prob3_reduced1, auc_only=False)
df_r1 = pd.DataFrame([df_en_r1, df1_r1, df2_r1, df3_r1])
df_r1.columns = ['AUCROC', 'AUCPR', 'ACC', 'BA', 'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK']
df_r1.to_csv("./outcomes/performance_r1.csv", index=None)

feature_importances_gb_r1  = clf1_reduced1.feature_importances_
feature_importances_rf_r1  = clf2_reduced1.feature_importances_
feature_importances_ert_r1 = clf3_reduced1.feature_importances_


attributes_r1 = list(indexes_to_attributes[i] for i in indexAttributes)


fea_im_pair_gb_r1 = sorted(zip(feature_importances_gb_r1, attributes_r1), reverse=True)
fea_im_gb_fullset_values_r1 = [fea_im_pair_gb_r1[i][0] for i in range(len(fea_im_pair_gb_r1))]
fea_im_gb_fullset_columns_r1 = [fea_im_pair_gb_r1[i][1] for i in range(len(fea_im_pair_gb_r1))]

fea_im_pair_rf_r1 = sorted(zip(feature_importances_rf_r1, attributes_r1), reverse=True)
fea_im_rf_fullset_values_r1 = [fea_im_pair_rf_r1[i][0] for i in range(len(fea_im_pair_rf_r1))]
fea_im_rf_fullset_columns_r1 = [fea_im_pair_rf_r1[i][1] for i in range(len(fea_im_pair_rf_r1))]

fea_im_pair_ert_r1 = sorted(zip(feature_importances_ert_r1, attributes_r1), reverse=True)
fea_im_ert_fullset_values_r1 = [fea_im_pair_ert_r1[i][0] for i in range(len(fea_im_pair_ert_r1))]
fea_im_ert_fullset_columns_r1 = [fea_im_pair_ert_r1[i][1] for i in range(len(fea_im_pair_ert_r1))]

fig, axs = plt.subplots(3, figsize=(15,10), tight_layout=True)

fea_im_fullset_columns_r1 = [fea_im_gb_fullset_columns_r1, fea_im_rf_fullset_columns_r1, fea_im_ert_fullset_columns_r1]
fea_im_fullset_values_r1 = [fea_im_gb_fullset_values_r1, fea_im_rf_fullset_values_r1, fea_im_ert_fullset_values_r1]
colors = ['C0', 'C1', 'C2']

# creating the bar plot
for i in range(3):
    axs[i].bar(fea_im_fullset_columns_r1[i], fea_im_fullset_values_r1[i], color=colors[i], width = 0.8)

for ax in axs.flat:
    ax.tick_params(axis='x', labelrotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set(xlabel='Features (Variables)', ylabel='Feature Importance', ylim =(0, 0.5))
    ax.xaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontsize(15)
    

fig.savefig("./outcomes/fea_im_r1.pdf")