# kaggle-diamonds

Kaggle competition to forecast the value of a diamond from their characteristics

![Found it](https://461v122bygqy1vf6uy3o7ubf-wpengine.netdna-ssl.com/wp-content/uploads/2020/01/Blood-Diamonds-2006.jpg)

---
## The Bases

The [bases](https://www.kaggle.com/c/diamonds-datamad0320) of the competition are:

*Goal*: Forecast the price of diamonds based on their characteristics.

*Evaluation criteria*: RMSE (Root Mean Squared Error).

*Deadline* : 12/05/2020 01:00 AM

*Results* :
All the process for the calculation of the 
- Train a minimum of 4 different models
- Perform a minimum of 4 Feature Extraction and Engineering techniques
- Documentation needed to reproduce the code
- The code in .py files that allows to reproduce the exercise.
- The Readme must contain a summary of the machine learning tools and algorithms and the results or the score obtained with each of them.

---
## The Data

The data supplied are three tables:

+ `diamonds_test.csv` -> 10 columns included index, three of the not numeric (13449)
+ `diamonds_train.csv` -> 11 columns included index, three of the not numeric (40345)
+ `sample_submision.csv` -> two columns, one id and two the price

The columns of the data are:

   -  carat `numeric` - unit of mass equal to 200 mg.
   -  cut    `categorical`  - Style and quality of the cutting, it affect the brillance
   -  color  `categorical`  - letter to label the color of the diamond from D to J 
   -  clarity `categorical` - visual appearance of internal characteristics of a diamond
   -  depth   `numeric`     - relation between the depth and the width 
   -  table  `numeric`  - refers to the flat facet of the diamond 
   -  x,y,z `numeric`     - dimensions 
   -  price `numeric`     

---
## Preprocess

> The preprocess were implemented in the data_cleaning.py and cleaning_functions.py
>
> The process can be called including the option `--data_c_t`

The preprocess of the data will include: dummies generation, categorial
generation, Normalization and filter the outlayers. 

The outlayers where found in variable `y`, in consequence two registers were excluded 

### Categorical and dummy generation

There are three variables categoricals, that were transform in dummy and
categorical form: cut, color and clarity.

The result of this part of the process were differents dataframes to be evaluated for the models. It includes the test data.

---
## The Models

>
> The models can be called with the option `--model` follow by the corresponding number 
>

The models used were:

- Linear: In the following graph you can see the performance of the model with two different input, the 
orange are encoding in integer the categorical variables, and the blue is converting this to dummies. 
![Result for linear](output/svr_lin_rmse_vs_sample_ne.png)

- SRV_rbf: 

The following models perform much better than the previuos.

- Random forest regession
- Gradient boosting regressor
- SGD Regressor 
- Histogram based gradient boosting regressor

![result](output/resume.png)

---
> The best model was `Histogram based gradient boosting regressor`. In consequence the tuning was made over it.

In the following graph it possible appreciate for which `learn_rate` the model perfom better:
