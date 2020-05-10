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
    diamonds_nor=diamonds_ne.reset_index().set_index('index')

    if model==1:
        mod.linear_SVR_dummies(diamonds_dum)
        mod.linear_SVR_ne(diamonds_ne)
    elif model==2:
#        mod.rbf_SVR_ne(diamonds_ne,0.35,deeper)
#        mod.rbf_SVR_no(diamonds_nor,diamonds_ne.price,0.35,deeper)
        mod.SVR_poly(diamonds_nor,diamonds_ne.price,0.35,deeper)
    elif model==3:
        # The parameter are:  Xy  and  test proportion of the sample
        mod2.hist_gra(diamonds_nor,0.35)
    elif model==4:
        # The parameter are:  Xy, test proportion of the sample and the
        # learning rate
        mod.gradient_reg(diamonds_nor,0.35,0.01)


