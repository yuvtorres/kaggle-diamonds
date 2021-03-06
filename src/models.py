import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt
import seaborn as sns

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def SVR_gen(diamonds,test_s,deeper,epsilon_v,C_v,type_dia):
    # receive the dataframe and the proportion of test/sample
    # diamonds:data , test_s: float (0-1) part of sample,
    # deeper: Bool if True makes 5 samples for each epsilon
    # if false makes just one, eps is a list of values for eps and C aswell.

    print(f'----- RBF SVR with normalize variables and eps={epsilon_v} -----')
    X=diamonds.drop(columns=['price'])
    if 'Unnamed: 0' in X.columns:
        X=X.drop(columns=['Unnamed: 0'])
    if 'level_0' in X.columns:
        X=X.drop(columns=['level_0'])

    y=diamonds.price

    MSE=[]
    x_test=[]
    y_test=[]

    for eps in epsilon_v:
        for c_val in C_v:
            dep=1
            svr = SVR(C=c_val , epsilon=eps)
            if deeper:
                k=5
            else:
                k=1
            while dep<=k:
                print(f"SVR with epsilon = {eps} and C = {c_val}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)
                y_predict=svr.fit(X_train,y_train).predict(X_test)
                MSE.append(mean_squared_error(y_predict,y_test))
                x_test.append(eps)
                #y_test.append(c_val)
                dep+=1

    print('The result of the model is in the ouput folder -> "svr_rbf_rmse_vs_epsilon_nor.png" ')
    plt.scatter(x_test,MSE)
    plt.xlabel('Epsilon')
    plt.ylabel('MSE value')
    plt.savefig('output/svr_rmse_vs_epsilon_'+type_dia+'.png')
    eps=sum(epsilon_v)/len(epsilon_v)
    c_val=sum(C_v)/len(C_v)
    print('Generating submission file ...')
    svr = SVR(C=c_val, epsilon=eps)
    svr.fit(X,y)
    X_test=pd.read_csv('output/diamonds_test_'+type_dia+'.csv')
    X_test=X_test.reset_index().set_index('index')
    if 'Unnamed: 0' in X_test.columns:
        X_test=X_test.drop(columns=['Unnamed: 0'])
    if 'level_0' in X_test.columns:
        X_test=X_test.drop(columns=['level_0'])
    try:
        y_sub=svr.predict(X_test)
    except:
        print('X_test:', X_test.columns,'-',X_test.shape)
        print('X:', X.columns,'-',X.shape)
    y_sub=pd.DataFrame({'id':range(len(y_sub)),'price': np.absolute(y_sub.astype(int))})

    y_sub.to_csv('output/svr_'+type_dia+'.csv',index=False)

    return True


def rbf_SVR_no(diamonds_nor,price,test_s,deeper):
    # receive the dataframe and the proportion of test/sample
    print('----- RBF SVR with kernel rbf and normalize variables  -----')
    X=diamonds_nor.drop(columns=['price'])
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
            RMSE.append(mean_squared_error(y_test,y_predict))
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
            RMSE.append(mean_squared_error(y_test,y_predict))
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
            RMSE.append(mean_squared_error(y_test,y_predict))
            x_test.append(test_s)


    print('The result of the model is in the ouput folder -> "svr_linear_r2_vs_sample_ne.png" ')
    plt.scatter(x_test,RMSE)
    plt.xlabel('Test proportion')
    plt.ylabel('RMSE value')
    plt.savefig('output/svr_lin_rmse_vs_sample_ne.png')

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
            RMSE.append(mean_squared_error(y_test,y_predict))
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

