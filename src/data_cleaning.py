import pandas as pd
import numpy as np
from sklearn import preprocessing as prp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Module for cleaning the data set of diamons.
def main():
    print('Importing the data ...')

# ***** Import the data ***** 
    diamonds_train=pd.read_csv('../data/diamonds_train.csv')
    diamonds_test=pd.read_csv('../data/diamonds_test.csv')
    print('Creating the index ...')

    # drop de 'Unnamed: 0' and reset index
    diamonds=diamonds_train.drop(columns=['Unnamed: 0'])
    diamonds=diamonds.reset_index().set_index('index')
    diamonds_test=diamonds_test.drop(columns=['Unnamed: 0'])
    diamonds_test=diamonds_test.reset_index().set_index('index')

    print('Saving correlation graph')    
    # Export the correlation
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(diamonds.corr(), annot=True, linewidths=.5, ax=ax)
    plt.savefig('../output/corr.png')

# ***** Creating the dummies ***** 
    print('Creating the dummies ...')
    dum_var=['cut','color','clarity']
    # Define the mat with dummies
    diamonds_dum=create_dummies(dum_var,diamonds)
    diamonds_test_dum=create_dummies(dum_var,diamonds_test)

    # Cleaning the diamons original from the dummies
    diamonds=diamonds.drop(columns=diamonds.columns[10:])
    print(diamonds.head())
    
    print('Saving Dummies ...')
    # Export the data_cleaning
    diamonds_dum.to_csv('../output/diamonds_dum.csv')
    diamonds_test_dum.to_csv('../output/diamonds_test_dum.csv')

# ***** Creating numeric encoding ***** 
    print('Creating numeric encoding')
    # Defining the label incoding
    diamonds_ne=create_numencod(diamonds,dum_var)
    diamonds_test_ne=create_numencod(diamonds_test,dum_var)
    
    print('Saving numeric encoding ...')
    # Export the data_cleaning
    diamonds_dum.to_csv('../output/diamonds_ne.csv')
    diamonds_test_dum.to_csv('../output/diamonds_test_ne.csv')

# ***** Creating PCA ***** 
    print('Creating PCA')
    # Defining the label incoding
    diamonds_PCA=create_PCA(diamonds,dum_var)
    diamonds_test_PCA=create_PCA(diamonds_test,dum_var)
    
    print('Saving PCA ...')
    # Export the data_cleaning
    diamonds_PCA.to_csv('../output/diamonds_PCA.csv')
    diamonds_test_PCA.to_csv('../output/diamonds_test_PCA.csv')




def create_PCA(diamonds,dum_var):
#functoin to create numerci encoding
    enc = prp.LabelEncoder()
    for var in dum_var:
        diamonds[var+'ne']=enc.fit_transform(diamonds[var])
    
    return diamonds.drop(columns=dum_var)


def create_numencod(diamonds,dum_var):
#function to create numeric encoding
    enc = prp.LabelEncoder()
    for var in dum_var:
        diamonds[var+'ne']=enc.fit_transform(diamonds[var])
    
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


if __name__ == "__main__": main()
