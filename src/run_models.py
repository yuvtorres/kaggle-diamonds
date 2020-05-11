import pandas as pd
import numpy as np
import src.models as mod
import src.models2 as mod2

import matplotlib.pyplot as plt
import seaborn as sns

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Module to run different models for data created in data_cleaning.py
def run_model(model, deeper=False):
    print('Importing diamonds_dum.csv')
    # import diamons_dum, drop unnamed and asign index
    diamonds_dum=pd.read_csv('output/diamonds_dum.csv')
    diamonds_dum=diamonds_dum.set_index('index')

    print('Importing diamonds_ne.csv')
    diamonds_ne=pd.read_csv('output/diamonds_ne.csv')
    diamonds_ne=diamonds_ne.set_index('index')

    print('Importing diamonds_nor.csv')
    diamonds_nor=pd.read_csv('output/diamonds_nor.csv')
    diamonds_nor=diamonds_nor.reset_index().set_index('index')

    print('Importing diamonds_PCA.csv')
    diamonds_PCA=pd.read_csv('output/diamonds_PCA.csv')
    diamonds_PCA=diamonds_PCA.reset_index().set_index('index')

    if model==1:
        mod.linear_SVR_dummies(diamonds_dum)
        mod.linear_SVR_ne(diamonds_ne)
    elif model==2:
#        mod.rbf_SVR_ne(diamonds_ne,0.35,deeper)
#        mod.rbf_SVR_no(diamonds_nor,diamonds_ne.price,0.35,deeper)
        mod.SVR_gen(diamonds_PCA,0.3,deeper,[0.7],[1.0],'PCA')
    elif model==3:
        # The parameter are:  Xy  and  test proportion of the sample
        mod2.rand_fores(diamonds_nor,0.3)
    elif model==4:
        # The parameter are:  Xy, test proportion of the sample and the
        # learning rate
        mod2.gradient_reg(diamonds_PCA,0.3,0.2)

def resume():
    print('Importing diamonds_dum.csv')
    # import diamons_dum, drop unnamed and asign index
    diamonds_dum=pd.read_csv('output/diamonds_dum.csv')
    diamonds_dum=diamonds_dum.set_index('index')

    print('Importing diamonds_ne.csv')
    diamonds_ne=pd.read_csv('output/diamonds_ne.csv')
    diamonds_ne=diamonds_ne.set_index('index')

    print('Importing diamonds_nor.csv')
    diamonds_nor=pd.read_csv('output/diamonds_nor.csv')
    diamonds_nor=diamonds_nor.reset_index().set_index('index')

    print('Importing diamonds_PCA.csv')
    diamonds_PCA=pd.read_csv('output/diamonds_PCA.csv')
    diamonds_PCA=diamonds_PCA.reset_index().set_index('index')

# ***** Variables setting **** model 2
    deeper=True
    epsilon_v=[ x/100.0 for x in range(40,200,20)]
    C_v=[1,2]
    mod.SVR_gen(diamonds_PCA,0.3,deeper,epsilon_v,C_v,'PCA')

# ***** model 3 
    mse1= mod2.rand_fores(diamonds_nor,0.3)
    mse2= mod2.rand_fores(diamonds_ne,0.3,'ne')
    mse3= mod2.rand_fores(diamonds_PCA,0.3,'PCA')
    mse4= mod2.rand_fores(diamonds_dum,0.3,'dum')
# **** model 4
    mse1= mod2.gradient_reg(diamonds_nor,0.3,0.2,'nor')
    mse2= mod2.gradient_reg(diamonds_ne,0.3,0.2,'ne')
    mse3= mod2.gradient_reg(diamonds_PCA,0.3,0.4,'PCA')
    mse4= mod2.gradient_reg(diamonds_dum,0.3,0.5,'dum')





    return True
