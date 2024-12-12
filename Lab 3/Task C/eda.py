import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

list_of_predictors = ["Age.1", "Temp", "OsSats", "Lympho", "WBC", "Plts", "Creatinine", "MAP", "Sodium", "ALT", "AST", "INR",
                      "BUN", "Troponin", "CrctProtein", "Ddimer", "Glucose", "Ferritin", "Procalcitonin", "IL6"]
                      
covid_19 = pd.read_excel("./data/covid_19.xlsx")
covid_19[list_of_predictors]

cohort0, cohort1 = {}, {}
for var in list_of_predictors:
    live = covid_19[(covid_19["Derivation cohort"]==0) & (covid_19[var]!=0.0) & (covid_19['Death']==0)][var].tolist()
    dead = covid_19[(covid_19["Derivation cohort"]==0) & (covid_19[var]!=0.0) & (covid_19['Death']==1)][var].tolist()
    cohort0[var] = (live, dead)
    
for var in list_of_predictors:
    live = covid_19[(covid_19["Derivation cohort"]==1) & (covid_19[var]!=0.0) & (covid_19['Death']==0)][var].tolist()
    dead = covid_19[(covid_19["Derivation cohort"]==1) & (covid_19[var]!=0.0) & (covid_19['Death']==1)][var].tolist()
    cohort1[var] = (live, dead)

stat_cohort0, stat_cohort1 = {}, {}
for predictor in list_of_predictors:
    mu_c0, mu_c1 = np.mean(cohort0[predictor][0]), np.mean(cohort1[predictor][1])
    sigma_c0, sigma_c1 = np.std(cohort0[predictor][0]), np.std(cohort1[predictor][1])
    n_c0, n_c1 = len(cohort0[predictor][0])+len(cohort0[predictor][1]), len(cohort1[predictor][0])+len(cohort1[predictor][1])
    stat_cohort0[predictor] = (n_c0, round(mu_c0, 2), round(sigma_c0,2))
    stat_cohort1[predictor] = (n_c1, round(mu_c1, 2), round(sigma_c1,2))

stat_cohort0_df = pd.DataFrame(stat_cohort0).T
stat_cohort0_df.columns = ['N', 'Mean', 'SD']
stat_cohort1_df = pd.DataFrame(stat_cohort1).T
stat_cohort1_df.columns = ['N', 'Mean', 'SD']
stat_cohort0_df.to_csv("./stat_cohort0_df.csv")
stat_cohort1_df.to_csv("./stat_cohort1_df.csv")

fig, axs = plt.subplots(5, 4, figsize=(25,25), tight_layout=True)

i, j = 0, 0
for predictor in list_of_predictors:
    mu0, mu1 = np.mean(cohort1[predictor][0]), np.mean(cohort1[predictor][1])
    sigma0, sigma1 = np.std(cohort1[predictor][0]), np.std(cohort1[predictor][1])
    num_bins=32
    _, bins0, patches = axs[i,j].hist(cohort1[predictor][0], num_bins, density=True, alpha=0.5, label="Alive")
    _, bins1, patches = axs[i,j].hist(cohort1[predictor][1], num_bins, density=True, alpha=0.5, label="Dead")
    y1 = ((1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * (1 / sigma1 * (bins0 - mu1))**2))
    y0 = ((1 / (np.sqrt(2 * np.pi) * sigma0)) * np.exp(-0.5 * (1 / sigma0 * (bins1 - mu0))**2))
    axs[i,j].plot(bins0, y0, '--', color="blue")
    axs[i,j].plot(bins1, y1, '--', color="orange")
    axs[i,j].set(xlabel=predictor, ylabel='Density')
    axs[i,j].legend(loc='upper right', fontsize=15)
    axs[i,j].xaxis.label.set_fontsize(20)
    axs[i,j].yaxis.label.set_fontsize(20)
    axs[i,j].tick_params(labelsize=20)
    if j <3:
        j=j+1
    elif j==3:
        i=i+1
        j=0
fig.savefig("./outcomes/hist_all.pdf")