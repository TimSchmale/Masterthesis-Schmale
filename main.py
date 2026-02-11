##### Script to perform the simulation study

import pandas as pd
from pathlib import Path
import random
from functions import get_leverage_scores, get_cross_leverage_scores, get_random_scores, get_combined_scores
import matplotlib.pyplot as plt

base = Path("/Users/timschmale/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studium/Data Science/Master/5. Semester/Masterarbeit/Masterthesis/Simulation Data")
folder = "rho0.30_p500_n100_k10"

## initialize lists to store data
df_train = []
df_test = []
y_train = []
y_test = []

## read in the simulation data
for i in range(5):
    df_train.append(
        pd.read_csv(f"{base}/{folder}/rep0{i+1}/X_train.csv")
    )
    df_test.append(
        pd.read_csv(f"{base}/{folder}/rep0{i+1}/X_test.csv")
    )
    y_train.append(
        pd.read_csv(f"{base}/{folder}/rep0{i+1}/y_train.csv")
    )
    y_test.append(
        pd.read_csv(f"{base}/{folder}/rep0{i+1}/y_test.csv")
    )

## set the seed
random.seed(1)

## set the basis parameters
k = 5
print(k)

## Calculate the scores per data frame
column_ls = []
column_cls = []
column_rs = []
column_cs = []

for i in range(5):
    column_ls.append(get_leverage_scores(df_train[i], k))
    column_cls.append(get_cross_leverage_scores(df_train[i], y_train[i]))
    column_rs.append(get_random_scores(df_train[i]))
    column_cs.append(get_combined_scores(df_train[i], y_train[i], k, p_leverage = 0.2))

## Investigation of distributions
plt.hist(column_ls[0], bins=100)
plt.show()
plt.hist(column_cls[0], bins=100)
plt.show()
plt.hist(column_rs[0], bins=100)
plt.show()
plt.hist(column_cs[0], bins=100)
plt.show()
