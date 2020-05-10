import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def gradient_reg(diamonds_nor,test_s,learn_rate):
    X=diamonds_nor.drop(columns=['price'])
    y=diamonds_nor['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)

    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': learn_rate , 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("For the GradientBoosting Regresor the MSE is: %.4f" % mse)

    print('Generating submission file ...')
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X, y)
    X_test=pd.read_csv('output/diamonds_test_nor.csv')
    X_test=X_test.reset_index().set_index('index')
    y_sub=regr.predict(X_test)
    y_sub=pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/gra_pred_nor.csv',index=False)

    return True

def hist_gra(diamonds_nor, test_s):
    X=diamonds_nor.drop(columns=['price'])
    y=diamonds_nor['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)

    reg_hist = HistGradientBoostingClassifier()
    reg_hist.fit(X_train, y_train)
    mse = mean_squared_error(y_test, reg_hist.predict(X_test))
    print("For the HistGradientBoostingClassifier Regresor the MSE is: %.4f" % mse)

    print('Generating submission file ...')
    reg_hist = HistGradientBoostingClassifier()
    reg_hist.fit(X, y)
    X_test = pd.read_csv('output/diamonds_test_nor.csv')
    X_test = X_test.reset_index().set_index('index')
    y_sub = regr.predict(X_test)
    y_sub = pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/hist_pred_nor.csv',index=False)

    return True


