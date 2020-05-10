# kaggle-diamonds

Kaggle competition to forecast the value of a diamond from their characteristics

![Found it](https://461v122bygqy1vf6uy3o7ubf-wpengine.netdna-ssl.com/wp-content/uploads/2020/01/Blood-Diamonds-2006.jpg)

---
## The Bases

The [bases](https://www.kaggle.com/c/diamonds-datamad0320) of the competition are:

*Goal*: Forecast the price of diamonds based on their characteristics.

*Evaluation criteria*: RMSE (Root Mean Squared Error).

*Deadline* : 12/05/2020 01:00 AM

*Requirements* :
- Train a minimum of 4 different models
- Perform a minimum of 4 Feature Extraction and Engineering techniques
- Documentation needed to reproduce the code
- The code in .py files that allows to reproduce the exercise.
- The Readme must contain a summary of the machine learning tools and algorithms and the results or the score obtained with each of them.

---
## The Data

The data supplied are three tables:

+ diamonds_test.csv -> 10 columns included index, three of the not numeric (13449)
+ diamonds_train.csv -> 11 columns included index, three of the not numeric (40345)
+ sample_submision.csv -> two columns, one id and two the price

The columns of the data are:

   -  carat:
   -  cut:
   -  color:
   -  clarity:
   -  depth:
   -  table:
   -  x,y,z:
   -  price:

---
## Preprocess

The preprocess of the data include: dummies generation, numeric encoding, PCA, and filter outlayers.

The result of this part of the process were differents dataframes to be evaluated for the models.

### Dummies generation and numeric encoding

The 

---
## The Algorithms

---
## Results
