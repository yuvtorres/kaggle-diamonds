import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def gradient_reg(diamonds,test_s,learn_rate,type_i='PCA'):
    X=diamonds.drop(columns=['price'])
    if 'Unnamed: 0' in X.columns:
        X=X.drop(columns=['Unnamed: 0'])
    if 'level_0' in X.columns:
        X=X.drop(columns=['level_0'])
    y=diamonds['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)

    params = {'n_estimators': 500, 'min_samples_split': 2,
          'learning_rate': learn_rate , 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("For the GradientBoosting Regresor the MSE is: %.4f" % mse)

    print('Generating submission file ...')
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X, y)
    X_test=pd.read_csv('output/diamonds_test_'+type_i+'.csv')
    X_test=X_test.reset_index().set_index('index')
    if 'Unnamed: 0' in X_test.columns:
        X_test=X_test.drop(columns=['Unnamed: 0'])
    if 'level_0' in X_test.columns:
        X_test=X_test.drop(columns=['level_0'])

    y_sub=clf.predict(X_test)
    y_sub=pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/pred_'+type_i+'.csv',index=False)

    return mse


def rand_fores(diamonds, test_s,type_i='nor'):
    X=diamonds.drop(columns=['price'])
    if 'Unnamed: 0' in X.columns:
        X=X.drop(columns=['Unnamed: 0'])
    if 'level_0' in X.columns:
        X=X.drop(columns=['level_0'])
    y=diamonds['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)

    reg = ensemble.RandomForestRegressor(bootstrap=False)
    reg.fit(X_train, y_train)
    mse = mean_squared_error(y_test, reg.predict(X_test))
    print("For the RandomForestRegressor the MSE is: %.4f" % mse)

    print('Generating submission file ...')
    reg = ensemble.RandomForestRegressor(bootstrap=False)
    reg.fit(X, y)
    X_test = pd.read_csv('output/diamonds_test_'+type_i+'.csv')
    X_test = X_test.reset_index().set_index('index')
    if 'Unnamed: 0' in X_test.columns:
        X_test=X_test.drop(columns=['Unnamed: 0'])
    if 'level_0' in X_test.columns:
        X_test=X_test.drop(columns=['level_0'])
    y_sub = reg.predict(X_test)
    y_sub = pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/pred_rafor_'+type_i+'.csv',index=False)

    return mse 


