import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import randint


X_train = np.load("./data/X_train.npy")
y_train = np.load("./data/y_train.npy")
k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

#numeric pipeline
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
#Categorical pipeline
bi_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
#ordinal pipeline
ord_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordputer", OrdinalEncoder())])

#KNN
knn_model =  Pipeline([("transformer", num_pipeline), ("classifier", KNeighborsClassifier(n_jobs=-1))]) 
knn_scores = cross_val_score(knn_model, X_train, y_train, scoring="roc_auc", cv=k_folds)
print(f"CV-mean: {round(knn_scores.mean(), 4)}, CV-sd: {round(knn_scores.std(), 4)}")

knn_param_distributions = {
    "classifier__n_neighbors": randint(3, 20),
    "classifier__p": [1, 2]
}

knn_random_search = RandomizedSearchCV(knn_model, knn_param_distributions, n_iter=300, cv=k_folds,
                                       n_jobs=-1, scoring="roc_auc", return_train_score=True, random_state=0)

start=time.time()
knn_random_search.fit(X_train, y_train)
print(f"CV: {round(knn_random_search.best_score_.mean(), 4)}")
knn_finalmodel = knn_model.set_params(**knn_random_search.best_params_)
knn_finalmodel.fit(X_train, y_train)
end=time.time()
print(f"Tuning time: {end-start}s")                             

knn_random_search.best_params_