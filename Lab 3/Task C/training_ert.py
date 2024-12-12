import time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import randint
from scipy.stats import uniform


X_train = np.load("./data/X_train.npy")
y_train = np.load("./data/y_train.npy")
k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

#ERT
ert_model = ExtraTreesClassifier(n_jobs=-1)
ert_scores = cross_val_score(ert_model, X_train, y_train, scoring="roc_auc", cv=k_folds)
print(f"CV-mean: {round(ert_scores.mean(), 4)}, CV-sd: {round(ert_scores.std(), 4)}")

ert_param_distributions = {
    "n_estimators": randint(100, 300),
    "max_features": uniform(0.6, 0.4),
    "max_depth": randint(2, 5),
    "min_samples_leaf": randint(1, 3),
}

ert_random_search = RandomizedSearchCV(ert_model, ert_param_distributions, n_iter=300, cv=k_folds,
                                       n_jobs=-1, scoring="roc_auc", return_train_score=True, random_state=0)

start=time.time()
ert_random_search.fit(X_train, y_train)
print(f"CV: {round(ert_random_search.best_score_.mean(), 4)}")
ert_finalmodel = ert_model.set_params(**ert_random_search.best_params_)
ert_finalmodel.fit(X_train, y_train)
end=time.time()
print(f"Tuning time: {end-start}s")                             

ert_random_search.best_params_