import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# Module to run different models for data created in data_cleaning.py
def main():
    # import diamons_dum, drop unnamed and asign index
    diamonds_dum=pd.read_csv('../output/diamonds_dum.csv')
    diamonds_dum=diamonds_dum.drop(columns=['Unnamed: 0']).set_index('indice')

    # Assing X (independient) and y (dependent) variables
    X=diamonds_dum.drop(columns=['price'])
    y=diamonds_dum.price

    # SVR with linear Kernel for different samples
    regr = LinearSVR(random_state=0, tol=1e-5, fit_intercept=True)
    test_sizes=[x / 100.0 for x in range(5, 50, 5)]
    r_2=[]

    for test_s in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)
        r_2.append(regr.fit(X_train, y_train).score(X_test, y_test))

    plt.scatter(test_sizes,r_2)
    plt.xlabel('Test proportion')
    plt.ylabel('R2 value')
    plt.savefig('../output/svr_liner_r2_vs_sample.png')


    pass

if __name__ == "__main__": main()
