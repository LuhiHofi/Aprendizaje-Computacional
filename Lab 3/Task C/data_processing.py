import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

covid_19 = pd.read_excel("./data/covid_19.xlsx")
covid_19["Derivation cohort"].value_counts()

missing_percentages = []
N = len(covid_19)

for c in covid_19.columns:
    missing_percentages.append(sum(covid_19[c] == 0) / N)

sorted(zip(missing_percentages, covid_19.columns), reverse=False)

list_of_predictors = ["Age.1", "Temp", "OsSats", "Lympho", "WBC", "Plts", "Creatinine", "MAP", "Sodium", "ALT", 
                     "AST", "INR", "BUN", "Troponin", "CrctProtein", "Ddimer", "Glucose", "Ferritin", "Procalcitonin", "IL6"]

X_train = covid_19[covid_19["Derivation cohort"]==1][list_of_predictors]
y_train = covid_19[covid_19["Derivation cohort"]==1]["Death"]
X_test = covid_19[covid_19["Derivation cohort"]==0][list_of_predictors]
y_test = covid_19[covid_19["Derivation cohort"]==0]["Death"]

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.25, random_state=42, stratify=y_test, shuffle=True)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
X_val  = X_val.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
y_val  = y_val.reset_index(drop=True)

np.save("./data/X_train.npy", X_train)
np.save("./data/X_test.npy", X_test)
np.save("./data/X_val.npy", X_val)

np.save("./data/y_train.npy", y_train)
np.save("./data/y_test.npy", y_test)
np.save("./data/y_val.npy", y_val)

