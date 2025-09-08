import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

def evaluate( x_test , t_test , model):
    pred_t = model.predict(x_test)
    rmse =np.sqrt( mean_squared_error(t_test,pred_t) )
    r2 = r2_score(t_test,pred_t)
    print(f'rmse is {rmse} -- and r2_score is {r2}')
    

def preparedata(train,test,processor = 0):
    train = train.to_numpy()
    test = test.to_numpy()
    x_train = train[:,:-1]
    t_train = train[:,-1]
    x_test = test[:,:-1]
    t_test = test[:,-1]
    if processor == 1:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif processor == 2:
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)


    return x_train , t_train , x_test , t_test

def approach1(x_train , t_train): 
    model = Ridge(fit_intercept=True,alpha=1)
    model.fit(x_train,t_train)
    return model

def approach2(x_train , t_train):
    model = XGBRegressor(
        n_estimators=300,     # number of boosting rounds
        learning_rate=0.1,    # step size
        max_depth=8,          # tree depth
        subsample=0.8,        # row sampling
        colsample_bytree=0.8, # feature sampling
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train,t_train)
    return model


def approach3(x_train , t_train):
    model = RandomForestRegressor(
        n_estimators=200,   # number of trees
        max_depth=20,       # limit depth to control overfitting (tune as needed)
        n_jobs=-1,          # use all CPU cores
        random_state=42
    )
    model.fit(x_train,t_train)
    return model 
    



if __name__ == '__main__':
    # I prepare this data using Jupyter , check Trip Duration EDA file
    train = pd.read_csv("df_train.csv")
    test = pd.read_csv("df_val.csv")
    x_train , t_train , x_test , t_test = preparedata(train,test,processor=1)
    model = approach2(x_train , t_train)
    evaluate( x_train , t_train ,model)
    evaluate( x_test , t_test ,model)



