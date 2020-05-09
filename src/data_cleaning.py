import pandas as pd
import numpy as np
from sklearn import preprocessing as prp
import matplotlib.pyplot as plt
import seaborn as sns


# Module for cleaning the data set of diamons.
def main():

    # Import the data
    diamonds=pd.read_csv('../data/diamonds_train.csv')

    # rename de 'Unnamed: 0' and id_diamond
    diamonds=diamonds.rename(columns={'Unnamed: 0':'indice'})
    diamonds.set_index('indice')

    # Creating the dummies
    dum_var=['cut','color','clarity']
    for var in dum_var:
        names=list(diamonds[var].value_counts().index)
        dum=prp.label_binarize(diamonds[var],names)
        dum=pd.DataFrame(dum)
        dum=dum.rename( columns=dict( zip( range(len(names)) ,[var+'_'+name for name in names] ) ) )
        dum['indice']=np.array(range(len(dum)))
        dum=dum.set_index('indice')
        diamonds=diamonds.join(dum,on='indice')

    # Define the mat with dummies
    diamonds_dum=diamonds.drop(columns=['cut','color','clarity'])

    # Cleaning the diamons original from the dummies
    diamonds=diamonds.drop(columns=diamonds.columns[10:])

    # Export the data_cleaning
    diamonds_dum.to_csv('../output/diamonds_dum.csv')

    # Export the correlation
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(diamonds.corr(), annot=True, linewidths=.5, ax=ax)
    plt.savefig('../output/corr.png')


if __name__ == "__main__": main()
