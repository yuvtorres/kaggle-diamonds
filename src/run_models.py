import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns


# Module to run different models for data created in data_cleaning.py
def main():
    linear_SVR_dummies()
        

def linear_SVR_dummies():
    print('Importing diamonds_dum.csv')
    # import diamons_dum, drop unnamed and asign index
    diamonds_dum=pd.read_csv('../output/diamonds_dum.csv')
    diamonds_dum=diamonds_dum.set_index('index')

    # Assing X (independient) and y (dependent) variables
    X=diamonds_dum.drop(columns=['price'])
    y=diamonds_dum.price

    # SVR with linear Kernel for different samples
    print('---- Testing the linear SVR')
    regr = LinearSVR(random_state=0, tol=1e-5, fit_intercept=True)
    test_sizes=[x / 100.0 for x in range(5, 50, 5)]
    r_2=[]

    for test_s in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)
        r_2.append(regr.fit(X_train, y_train).score(X_test, y_test))


    print('The result of the model is in the ouput folder ->  "svr_liner_r2_vs_sample.png" ')
    plt.scatter(test_sizes,r_2)
    plt.xlabel('Test proportion')
    plt.ylabel('R2 value')
    plt.savefig('../output/svr_liner_r2_vs_sample.png')
    
    print('Generating submission file ...')
    
    X_test=pd.read_csv('../output/diamonds_test_dum.csv')
    X_test=X_test.set_index('index')
    y_sub=regr.predict(X_test)
    y_sub=pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})
    
    y_sub.to_csv('../output/linear_pred.csv',index=False)

    


if __name__ == "__main__": main()
