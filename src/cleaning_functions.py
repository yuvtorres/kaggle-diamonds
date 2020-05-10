import pandas as pd
import numpy as np
from sklearn import preprocessing as prp
from sklearn.decomposition import PCA

# Module for cleaning the data set of diamonds.

def normalize(diamonds,dum_var):
    # normalize the data using quantile transform
    quant_t=prp.QuantileTransformer( output_distribution='normal',
            copy=False)
    names_col=list(diamonds.columns)
    diamonds_nor=quant_t.fit_transform(diamonds)
    diamonds_nor=pd.DataFrame(dict(zip(names_col,diamonds_nor)))
    return diamonds_nor


def create_PCA(diamonds,dum_var):
#function to create numeric encoding
    pca = PCA(n_components=1)
    diamonds['pca_dum']=pca.fit_transform(diamonds[dum_var].to_numpy())
    return diamonds.drop(columns=dum_var)


def create_numencod(diamonds,dum_var):
#function to create numeric encoding
    enc = prp.LabelEncoder()
    for var in dum_var:
        diamonds[var+'_ne']=enc.fit_transform(diamonds[var])

    return diamonds.drop(columns=dum_var)

def create_dummies(dum_var,diamonds):
# function to create dummies variables
    for var in dum_var:
        names=list(diamonds[var].value_counts().index)
        dum=prp.label_binarize(diamonds[var],names)
        dum=pd.DataFrame(dum)
        dum=dum.rename( columns=dict( zip( range(len(names)) ,[var+'_'+name for name in names] ) ) )
        dum['index']=np.array(range(len(dum)))
        dum=dum.set_index('index')
        diamonds=diamonds.join(dum,on='index')

    return diamonds.drop(columns=dum_var)

