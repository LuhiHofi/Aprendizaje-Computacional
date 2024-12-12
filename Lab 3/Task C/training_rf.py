import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import randint
from scipy.stats import uniform


X_train = np.load("./data/X_train.npy")
y_train = np.load("./data/y_train.npy")
k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

#RF
rf_model = RandomForestClassifier(n_jobs=-1)
rf_scores = cross_val_score(rf_model, X_train, y_train, scoring="roc_auc", cv=k_folds)
print(f"CV-mean: {round(rf_scores.mean(), 4)}, CV-sd: {round(rf_scores.std(), 4)}")

rf_param_distributions = {
    "n_estimators": randint(100, 300),
    "max_features": uniform(0.6, 0.4),
    "max_depth": randint(2, 5),
    "min_samples_leaf": randint(1, 3),
}

rf_random_search = RandomizedSearchCV(rf_model, rf_param_distributions, n_iter=300, cv=k_folds,
                                      n_jobs=-1, scoring="roc_auc", return_train_score=True, random_state=0)

start=time.time()
rf_random_search.fit(X_train, y_train)
print(f"CV: {round(rf_random_search.best_score_.mean(), 4)}")
rf_finalmodel = rf_model.set_params(**rf_random_search.best_params_)
rf_finalmodel.fit(X_train, y_train)
end=time.time()
print(f"Tuning time: {end-start}s")                                   

rf_random_search.best_params_