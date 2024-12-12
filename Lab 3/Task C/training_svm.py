import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score


X_train = np.load("./data/X_train.npy")
y_train = np.load("./data/y_train.npy")
k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

#numeric pipeline
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
#Categorical pipeline
bi_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
#ordinal pipeline
ord_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordputer", OrdinalEncoder())])

#SVM
svm_model = Pipeline([("transformer", num_pipeline), ("classifier", SVC())])
svm_scores = cross_val_score(svm_model, X_train, y_train, scoring="roc_auc", cv=k_folds)
print(f"CV-mean: {round(svm_scores.mean(), 4)}, CV-sd: {round(svm_scores.std(), 4)}")

svm_param_distributions = {
    "classifier__C": [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100, 1000],
    "classifier__gamma": [0.001, 0.005, 0.01, 0.05, 0.1, 1, 10, 100, 1000]
}

svm_random_search = RandomizedSearchCV(svm_model, svm_param_distributions, n_iter=300, cv=k_folds,
                                       n_jobs=-1, scoring="roc_auc", return_train_score=True, random_state=0)

start=time.time()
svm_random_search.fit(X_train, y_train)
print(f"CV: {round(svm_random_search.best_score_.mean(), 4)}")
svm_finalmodel = svm_model.set_params(**svm_random_search.best_params_)
svm_finalmodel.fit(X_train, y_train)
end=time.time()
print(f"Tuning time: {end-start}s")                      

svm_random_search.best_params_