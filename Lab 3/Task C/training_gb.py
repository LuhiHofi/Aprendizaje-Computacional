import time
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import randint
from scipy.stats import uniform


X_train = np.load("./data/X_train.npy")
y_train = np.load("./data/y_train.npy")
k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

#GB
gb_model = GradientBoostingClassifier(random_state=0)
gb_scores = cross_val_score(gb_model, X_train, y_train, scoring="roc_auc", cv=k_folds)
print(f"CV-mean: {round(gb_scores.mean(), 4)}, CV-sd: {round(gb_scores.std(), 4)}")

gb_param_distributions = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": randint(100, 300),
    "subsample": uniform(0.6, 0.4),
    "max_depth": randint(2, 5),
    "min_samples_leaf": randint(1, 3),
}

gb_random_search = RandomizedSearchCV(gb_model, gb_param_distributions, n_iter=300, cv=k_folds, 
                                      n_jobs=-1, scoring="roc_auc", return_train_score=True, random_state=0)

start=time.time()
gb_random_search.fit(X_train, y_train)
print(f"CV: {round(gb_random_search.best_score_.mean(), 4)}")
gb_finalmodel = gb_model.set_params(**gb_random_search.best_params_)
gb_finalmodel.fit(X_train, y_train)
end=time.time()
print(f"Tuning time: {end-start}s")                                      

gb_random_search.best_params_