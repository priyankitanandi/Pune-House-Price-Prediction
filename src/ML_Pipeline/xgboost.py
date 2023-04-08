from scipy.stats import uniform
from sklearn.model_selection import ParameterSampler
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from ML_Pipeline.model_evaluation import plotResidue
from sklearn.model_selection import KFold


## function to generate the parameters list for xgboost model
def hyperparameter_xgb():
    param_grid = {'max_depth':range(2,7), # define the parameters for XGBoost model
        'gamma':uniform(loc=0.0, scale=3), # minimum of the loss function rduction to split a node
        'min_child_weight':range(3,6), # similar to min_samples_leaf
        'colsample_bytree':uniform(loc=0.1, scale=0.9), # similar to the max_features
        'subsample':uniform(loc=0.5, scale=0.5), # similar to bootstraping in RF
        'learning_rate':uniform(loc=0.01, scale=0.99)} # contriburion rate of each estimator

    rng = np.random.RandomState(20)
    n_iter=500
    param_list = list(ParameterSampler(param_grid, n_iter=n_iter,
                                    random_state=rng))
    return param_list


param_list = hyperparameter_xgb()


# function to perform cross validation for xgboost model
def cross_validate( est, Xn, yn, n_fold=10 ):
    """
    Cross validation for XGB fit.
    Params:
        est: xgb regressor
        Xn: numpy array (n_sample, n_feature)
            Training feature matrix
        yn: numpy array (n_sample,)
            Training target vector
        n_fold: int
            number of folds for cross validating
    """
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=0)
    mean_train_error, mean_val_error = 0., 0.
    for train_index, val_index in kf.split(Xn, yn):
        est.fit(Xn[train_index], yn[train_index], 
                    eval_set=[(Xn[train_index], yn[train_index]), 
                              (Xn[val_index], yn[val_index])],
                    eval_metric='rmse',
                    verbose=False, 
                    early_stopping_rounds=30)
        mean_val_error += mean_squared_error(yn[val_index], est.predict(Xn[val_index]))
        mean_train_error += mean_squared_error(yn[train_index], est.predict(Xn[train_index]))

    return mean_train_error/n_fold, mean_val_error/n_fold



## function to xgboost model
def xgboost_model(X_train,y_train,X_test,y_test,X,y,rs):
    xgbr = xgb.XGBRegressor( objective='reg:squarederror', n_estimators=1000, verbosity=1 ) # XGBoost model

    val_score, train_score = [], []
    counter = 0
    for param_grid in param_list:
        xgbr.set_params(**param_grid)
        train_error, val_error = cross_validate(xgbr, X_train.values, y_train.values)  
        val_score.append(val_error)
        train_score.append(train_error)
    
        if counter%50 == 0 :
            print("iter =", counter, 
                "train_score=", train_score[counter], ", ", 
                "val_score=", val_score[counter])
        counter += 1
        
    # create a dataframe
    df_grid = pd.DataFrame(param_list)
    df_grid["train_score"] = train_score
    df_grid["val_score"] = val_score

    # check for the best grid value
    df_grid_best = df_grid[ df_grid["val_score"]==min(val_score) ]

    # best parameters
    best_params = df_grid_best.iloc[0, :-2].to_dict()
    best_params["max_depth"] = int(best_params["max_depth"])

    # fit the model
    xgbr.set_params(**best_params)
    xgbr.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric='rmse',
                    verbose=True, 
                    early_stopping_rounds=30)

    xgbr.get_booster().attributes()

    #plot the residuals
    plotResidue(xgbr, X, y, rs=rs)
    print (xgbr.get_booster().attributes())

 