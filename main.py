import joblib
import numpy as np
import pandas as pd
import os
import argparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from datetime import datetime
import json
import warnings

from sklearn.model_selection import KFold, cross_val_score
from feature_engineering import *
warnings.filterwarnings("ignore")

'''
features can be used 
vendor_id,pickup_month,pickup_hour,week_day,passenger_count,pickup_longitude,pickup_latitude
,dropoff_longitude,dropoff_latitude,long_distance,lat_distance,cluster_pair_avg_duration,
is_weekend,rush_hour
'''

np.random.seed(42)

def evaluate( x_test , t_test , model):
    pred_t = model.predict(x_test)
    rmse =np.sqrt( mean_squared_error(t_test,pred_t) )
    r2 = r2_score(t_test,pred_t)
    print(f'rmse is {rmse} -- and r2_score is {r2}')
    return {'r2':r2,'rmse':rmse}


def save_run(model,args=None,metrics=None,imputer=None,encoder=None,scaler=None,base_dir='model_utilities'):
    """
    Save a full ML run in a unique folder, using  args, model, metrics, and optionally x_train.

    Parameters:
        model      : Trained model object
        args       : argparse.Namespace with configuration
        metrics    : Optional dict of metrics (e.g., {"r2": 0.85,"rmse": 2.3})
        base_dir   : Base directory to store the latest run
    """
    # ensure base_dir exist
    os.makedirs(base_dir,exist_ok=True)
    # save the model,imputer,scaler,encoder
    mode_dict = {
        'model':model,
        'imputer':imputer,
        'encoder':encoder,
        'scaler':scaler
    }
    joblib.dump(mode_dict,os.path.join(base_dir,'model_dict.pkl'))
    # save args
    if args is not None:
        args_dict = vars(args)
        with open(os.path.join(base_dir,'args.json'),'w') as f:
            json.dump(args_dict,f,indent=4)
    # save metrics
    if metrics is not None:
        with open(os.path.join(base_dir,'metrics.json'),'w') as f:
            json.dump(metrics,f,indent=4)
    print(f"✨ Run saved successfully at: {base_dir}")



def load_data(args):
    # reading train and val csv files
    df_train = pd.read_csv(args.train_path)
    df_val = pd.read_csv(args.val_path)
    # delete id and store_and_fwd_flag columns
    df_train.drop(columns=['id','store_and_fwd_flag'],inplace=True)
    df_val.drop(columns=['id','store_and_fwd_flag'],inplace=True)
    
    # delete outliers
    if(args.del_outliers):
        df_train = del_outliers(df_train,args.outliers_factor)
    
    # filter the data
    df_train,df_val = filter_data(df_train,df_val)

 
    # prepare date columns
    df_train,df_val = prepare_date_columns(df_train,df_val)

    # add rush_hour feature
    if(args.rush_hour):
        df_train,df_val = add_rush_hour(df_train,df_val)

    # add is_weekend feature
    if(args.is_weekend):
        df_train,df_val = add_is_weekend(df_train,df_val)

    # add distance features
    if(args.long_lat_distance):
        df_train,df_val = add_long_lat_distance(df_train,df_val)
    
    df_train['haversine_distance'] = haversine(
    df_train['pickup_latitude'].values,
    df_train['pickup_longitude'].values,
    df_train['dropoff_latitude'].values,
    df_train['dropoff_longitude'].values)
    
    df_val['haversine_distance'] = haversine(
    df_val['pickup_latitude'].values,
    df_val['pickup_longitude'].values,
    df_val['dropoff_latitude'].values,
    df_val['dropoff_longitude'].values)


    # add cluster_pair_avg_duration
    imputer = None
    if(args.cluster_pair_avg_duration):
        df_train,df_val,imputer = add_cluster_pair_avg_duration(df_train,df_val,args.imputer)
    
    # split data
    x_train,x_val,t_train,t_val=split_data(df_train,df_val)

 
    # Encodeing Categorical Data
    encoder = None
    if(args.encode):
        x_train,x_val,encoder = encode(x_train,x_val,args.encode)
    
    # scaling
    scaler = None
    if(args.scaler):
        x_train,x_val,scaler = scaling(x_train,x_val,args.scaler)
    
    # Log Target
    if(args.log_target):
        t_train = np.log1p(t_train)
        t_val = np.log1p(t_val)
    
    return x_train,x_val,t_train,t_val,imputer,encoder,scaler
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training And Validation Model')
    parser.add_argument('--is_weekend',action='store_true',
                        help='This Feature determines if the day is weekend or not')
    parser.add_argument('--rush_hour',action='store_true',
                        help='This Feature determines the rush hours of day')
    parser.add_argument('--cluster_pair_avg_duration',action='store_true',
                        help='This Feature compute the average trip duration per every pair of clusters')
    parser.add_argument('--scaler',type=str,choices=['standardscaler','minmaxscaler'],
                        default='standardscaler',help='Scaling Method')
    parser.add_argument('--del_outliers',action='store_true',
                        help='Delete outliers using IQR method(you can determine the factor or keep it 1.5)')
    parser.add_argument('--outliers_factor',type=float,default=1.5,
                        help='when factor increase the number of deleted outliers decrease and vice versa')
    parser.add_argument('--long_lat_distance',action='store_true',
                        help='This Feature compute longitude_distance and latitude_distance of the trips')
    parser.add_argument('--encode',type=str,nargs='+',choices=['passenger_count' ,'pickup_month','pickup_weekday','pickup_hour','vendor_id','rush_hour','is_weekend'],
                        help='encode Features using One-Hot_encoder, features allowed :(passenger_count ,pickup_month,pickup_weekday,pickup_hour,vendor_id,rush_hour,is_weekend)')
    parser.add_argument('--log_target',action='store_true',help='use log target')
    parser.add_argument("--train_path", type=str, default=r"split\train.csv",
                        help="Path to training CSV file.")
    parser.add_argument("--val_path", type=str, default=r"split\val.csv",
                        help="Path to val CSV file.")
    parser.add_argument("--imputer", type=str, default="knn",choices=['knn','simple','iterative'],
                        help="To impute the missing values resulted from avg_duration feature")
    args = parser.parse_args()

    x_train,x_val,t_train,t_val,imputer,encoder,scaler = load_data(args)
    print(f'features used {x_train.columns}')
    # train the model ( only ridge model with alpha 1 is permitted )        
    model = Ridge(alpha=1,random_state=42)
    model.fit(x_train,t_train)
    metrics = evaluate(x_val,t_val,model)
    # save_run(model,args,metrics,imputer,encoder,scaler)

