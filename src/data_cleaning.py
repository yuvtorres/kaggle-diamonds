import pandas as pd
import numpy as np
import src.cleaning_functions as c_f

import matplotlib.pyplot as plt
import seaborn as sns

# Module for cleaning the data and make some graphs.

def pre_graph(graph):
    # 1 for corr, 2 for matrix and 3 for both

    diamonds=pd.read_csv('data/diamonds_train.csv')
    if graph==1 or graph==3:
# ***** Creating the correlation graph ***** 
   # Export the correlation
        print('Saving correlation graph...')
        f, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(diamonds.corr(), annot=True, linewidths=.5, ax=ax)
        plt.savefig('output/corr.png')

    if graph==2 or graph==3:
# ***** Creating the matrix graph ***** 
        print('Saving matrix graph...')
        g = sns.PairGrid(diamonds[['carat','depth','table','x','y','z','price']])
        g.map_diag(plt.hist)
        g.map_offdiag(plt.scatter);
        plt.savefig('output/descrip_var.png')




def cleaning_data():
    print('Importing the data ...')

# ***** Import the data ***** 
    diamonds_train=pd.read_csv('data/diamonds_train.csv')
    diamonds_test=pd.read_csv('data/diamonds_test.csv')

    # ***** Take out the outlayer ***** 
    diamonds_train=diamonds_train.loc[diamonds_train.y<30]

    print('Creating the index ...')
    # drop de 'Unnamed: 0' and reset index
    diamonds=diamonds_train.drop(columns=['Unnamed: 0'])
    diamonds['index']=range(len(diamonds))
    diamonds=diamonds.set_index('index')
    diamonds_test=diamonds_test.drop(columns=['Unnamed: 0'])
    diamonds_test=diamonds_test.reset_index().set_index('index')
# ***** Creating the dummies ***** 
    print('Creating the dummies ...')
    dum_var=['cut','color','clarity']
    # Define the mat with dummies
    diamonds_dum=c_f.create_dummies(dum_var,diamonds)
    diamonds_test_dum=c_f.create_dummies(dum_var,diamonds_test)

    # Cleaning the diamonds original from the dummies
    diamonds=diamonds.drop(columns=diamonds.columns[10:])


    print('Saving Dummies ...')
    # Export the data_cleaning
    diamonds_dum.to_csv('output/diamonds_dum.csv')
    diamonds_test_dum.to_csv('output/diamonds_test_dum.csv')

# ***** Creating numeric encoding ***** 
    print('Creating numeric encoding ...')
    # Defining the label incoding
    diamonds_ne=c_f.create_numencod(diamonds.drop(columns=['price']),dum_var)
    diamonds_test_ne=c_f.create_numencod(diamonds_test,dum_var)
    diamonds_ne['price']=diamonds['price']


    print('Saving numeric encoding ...')
    # Export the data_cleaning
    diamonds_ne.to_csv('output/diamonds_ne.csv')
    diamonds_test_ne.to_csv('output/diamonds_test_ne.csv')

# ***** Creating PCA ***** 
    print('Creating PCA ...')
    # Creating PCA
    dum_var_ne=[var+'_ne' for var in dum_var]
    diamonds_PCA=c_f.create_PCA(diamonds_ne,dum_var_ne)
    diamonds_test_PCA=c_f.create_PCA(diamonds_test_ne,dum_var_ne)
    diamonds_PCA['price']=diamonds['price']

    print('Saving PCA ...')
    # Export the data_cleaning
    diamonds_PCA.to_csv('output/diamonds_PCA.csv')
    diamonds_test_PCA.to_csv('output/diamonds_test_PCA.csv')

# ***** Creating Normalize *****
    print('Normalizing the data... ')
    # Normalizing
    diamonds_nor=c_f.normalize(diamonds_ne,dum_var)
    diamonds_test_nor=c_f.normalize(diamonds_test_ne,dum_var)
    diamonds_nor['price']=diamonds['price']

    print('Saving  normalize...')
    # Export the data_cleaning
    diamonds_nor.to_csv('output/diamonds_nor.csv')
    diamonds_test_nor.to_csv('output/diamonds_test_nor.csv')

