import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def SVR_poly(diamonds_nor,price,test_s,deeper):
    # receive the dataframe and the proportion of test/sample
    # diamonds_nor: X normalized, price: vector of prices, test_s: float (0-1)
    # part of sample, deeper: Bool if True makes 5 samples for each epsilon
    # if false makes just one

    print('----- RBF SVR with polynomic kernel and normalize variables  -----')
    X=diamonds_nor
    y=price

    # SVR with polynomical Kernel for different samples
    print('---- Testing the RBF polynomic kernel with params gamma=0.1 and epsilon [0.5 , 1.5 , 2.5]')

    epsilon_v =[0.5,1.5,2.5]
    RMSE=[]
    x_test=[]

    for eps in epsilon_v:
        print(f"RBF Polynomic with epsilon = {eps}")
        dep=1
        svr_poly = SVR(kernel='poly',C=100 , gamma='auto', degree=2, coef0=1, epsilon=eps)
        if deeper:
            k=5
        else:
            k=1
        while dep<=k:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)
            y_predict=svr_poly.fit(X_train,y_train).predict(X_test)
            RMSE.append((((y_predict-y_test)**2).sum()/len(y_test))**0.5)
            x_test.append(eps)
            dep+=1

    print('The result of the model is in the ouput folder -> "svr_rbf_rmse_vs_epsilon_nor.png" ')
    plt.scatter(x_test,RMSE)
    plt.xlabel('Epsilon')
    plt.ylabel('RMSE value')
    plt.savefig('output/svr_poly_rmse_vs_epsilon_nor.png')

    print('Generating submission file ...')
    svr_poly = SVR(kernel='poly',C=100 , gamma=auto, degree=2, coef0=1, epsilon=eps)
    svr_poly.fit(X,y)
    X_test=pd.read_csv('output/diamonds_test_nor.csv')
    X_test=X_test.reset_index().set_index('index')
    y_sub=regr.predict(X_test)
    y_sub=pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/poly_pred_nor.csv',index=False)

    return True


def rbf_SVR_no(diamonds_nor,price,test_s,deeper):
    # receive the dataframe and the proportion of test/sample
    print('----- RBF SVR with kernel rbf and normalize variables  -----')
    X=diamonds_nor
    y=price

    # SVR with RBF Kernel for different samples
    print('---- Testing the RBF SVR with params gamma=0.1 and epsilon [0.5 , 1.5 , 2.5]')

    epsilon_v =[0.5,1.5,2.5]
    RMSE=[]
    x_test=[]

    for eps in epsilon_v:
        dep=1
        regr = SVR(kernel='rbf',C=100 , gamma=0.1, epsilon=eps)
        if deeper: 
            k=5 
        else:
            k=1
        while dep<=k:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)
            y_predict=regr.fit(X_train,y_train).predict(X_test)
            RMSE.append((((y_predict-y_test)**2).sum()/len(y_test))**0.5)
            x_test.append(eps)
            dep+=1

    print('The result of the model is in the ouput folder -> "svr_rbf_rmse_vs_epsilon_nor.png" ')
    plt.scatter(x_test,RMSE)
    plt.xlabel('Epsilon')
    plt.ylabel('R2 value')
    plt.savefig('output/svr_rbf_r2_vs_epsilon_nor.png')

    print('Generating submission file ...')
    regr = SVR(kernel='rbf',C=100 , gamma=0.1, epsilon=0.7)
    regr.fit(X,y)
    X_test=pd.read_csv('output/diamonds_test_nor.csv')
    X_test=X_test.reset_index().set_index('index')
    y_sub=regr.predict(X_test)
    y_sub=pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/rbf_pred_nor.csv',index=False)


    return True


def rbf_SVR_ne(diamonds_ne,test_s,deeper):
    # receive the dataframe and the proportion of test/sample
    print('----- RBF SVR with numeric enconder  -----')
    X=diamonds_ne.drop(columns=['price'])
    y=diamonds_ne.price

    # SVR with linear Kernel for different samples
    print('---- Testing the RBF SVR with params gamma=0.1 and epsilon [0.5 , 1.5 , 2.5]')

    epsilon_v =[0.5,1.5,2.5]
    RMSE=[]
    x_test=[]

    for eps in epsilon_v:
        dep=1
        if deeper:
            k=5
        else:
            k=1

        regr = SVR(kernel='rbf',C=100 , gamma=0.1, epsilon=eps)
        while dep<=k:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)
            y_predict=regr.fit(X_train,y_train).predict(X_test)
            RMSE.append((((y_predict-y_test)**2).sum()/len(y_test))**0.5)
            x_test.append(eps)
            dep+=1

    print('The result of the model is in the ouput folder -> "svr_rbf_rmse_vs_epsilon_ne.png" ')
    plt.scatter(x_test,RMSE)
    plt.xlabel('Epsilon')
    plt.ylabel('R2 value')
    plt.savefig('output/svr_rbf_r2_vs_epsilon_ne.png')

    print('Generating submission file ...')
    regr = SVR(kernel='rbf',C=100 , gamma=0.1, epsilon=0.7)
    regr.fit(X,y)
    
    X_test=pd.read_csv('output/diamonds_test_ne.csv')
    X_test=X_test.set_index('index')
    y_sub=regr.predict(X_test)
    y_sub=pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/rbf_pred.csv',index=False)


def linear_SVR_ne(diamonds_ne):
    print('-----  Linear SVR with numeric enconder  -----')
    X=diamonds_ne.drop(columns=['price'])
    y=diamonds_ne.price

    # SVR with linear Kernel for different samples
    print('---- Testing the linear SVR')
    regr = LinearSVR(random_state=0, tol=1e-5, fit_intercept=True)
    test_sizes=[x / 100.0 for x in range(5, 50, 5)]
    RMSE=[]
    x_test=[]

    for test_s in test_sizes:
        for k in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)
            y_predict=regr.fit(X_train,y_train).predict(X_test)
            RMSE.append((((y_predict-y_test)**2).sum()/len(y_test))**0.5)
            x_test.append(test_s)


    print('The result of the model is in the ouput folder -> "svr_linear_r2_vs_sample_ne.png" ')
    plt.scatter(x_test,RMSE)
    plt.xlabel('Test proportion')
    plt.ylabel('R2 value')
    plt.savefig('output/svr_liner_r2_vs_sample_ne.png')

    print('Generating submission file ...')

    X_test=pd.read_csv('output/diamonds_test_ne.csv')
    X_test=X_test.set_index('index')
    y_sub=regr.predict(X_test)
    y_sub=pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/linear_pred.csv',index=False)


def linear_SVR_dummies(diamonds_dum):
    print('-----  Linear SVR with dummies  -----')
    # Assing X (independient) and y (dependent) variables
    X=diamonds_dum.drop(columns=['price'])
    y=diamonds_dum.price

    # SVR with linear Kernel for different samples
    print('---- Testing the linear SVR with dummies')
    regr = LinearSVR(random_state=0, tol=1e-5, fit_intercept=True)
    test_sizes=[x / 100.0 for x in range(5, 50, 5)]
    RMSE=[]
    x_size=[]

    for test_s in test_sizes:
        for k in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)
            y_predict=regr.fit(X_train,y_train).predict(X_test)
            RMSE.append((((y_predict-y_test)**2).sum()/len(y_test))**0.5)
            x_size.append(test_s)

    print('The result of the model is in the ouput folder ->  "svr_linear_rmse_vs_sample.png" ')
    plt.scatter(x_size,RMSE,s=1.5)
    plt.xlabel('Test proportion')
    plt.ylabel('RMSE value')
    plt.title('Linear SVR')
    plt.savefig('output/svr_linear_rmse_vs_sample.png')
    
    print('Generating submission file ...')
    
    X_test=pd.read_csv('output/diamonds_test_dum.csv')
    X_test=X_test.set_index('index')
    y_sub=regr.predict(X_test)
    y_sub=pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/linear_pred.csv',index=False)

