from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import pandas as pd
import h2o
def h2o_grid():
    h2o.init()
    data = h2o.import_file('output/diamonds_PCA.csv')
    splits = data.split_frame(ratios=[0.7, 0.15], seed=1)
    train = splits[0]
    valid = splits[1]
    test = splits[2]
    y = 'price'
    x = list(data.columns)

    x.remove(y)

    hyper_parameters = {'learn_rate': [0.01, 0.1],'max_depth': [3, 5, 9],
            'sample_rate': [0.8, 1.0],'col_sample_rate': [0.2, 0.5, 1.0]}

    gs = H2OGridSearch(H2OGradientBoostingEstimator,hyper_parameters)

    gs.train(x = x,y=y, training_frame=train,validation_frame=valid)
    gs1=gs.get_grid(sort_by='rmse',decreasing=True)
    best_m=gs1.models[0]
    best_mp=best_m.model_performance(test)
    print(best_mp.rmse())
    test = h2o.import_file('output/diamonds_test_PCA.csv')
    predict=best_m.predict(test)
    predict=h2o.as_list(predict) 
    predict.to_csv('output/pred_h2o.csv') 


