import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from feature_engineering import haversine
def filter_data(df_test):
    # handle trips out of New York
    min_lat = 40.49
    max_lat = 40.92
    min_lon = -74.27
    max_lon = -73.68
    df_test = df_test[(df_test['pickup_latitude'].between(min_lat,max_lat))& 
                        (df_test['pickup_longitude'].between(min_lon,max_lon))&
                        (df_test['dropoff_latitude'].between(min_lat,max_lat))& 
                        (df_test['dropoff_longitude'].between(min_lon,max_lon))]
    # delete passenger_count values 0 and 7
    upper , lower = 7 , 0
    df_test = df_test[(df_test['passenger_count'] > lower) & (df_test['passenger_count'] < upper)]
    return df_test


def prepare_date_columns(df_test):
    df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])

    df_test['pickup_month'] = df_test['pickup_datetime'].dt.month

    df_test['pickup_hour'] = df_test['pickup_datetime'].dt.hour

    df_test['pickup_weekday'] = df_test['pickup_datetime'].dt.weekday
    
    # delete date column
    df_test = df_test.drop(columns=['pickup_datetime'])
    return df_test

def add_rush_hour(df_test):
    df_test['rush_hour'] = df_test['pickup_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
    return df_test

def add_is_weekend(df_test):
    df_test['is_weekend'] = df_test['pickup_weekday'].isin([5,6]).astype(int) # saturday and sunday
    return df_test

def add_long_lat_distance(df_test):
    df_test["latitude_distance"]  = df_test["dropoff_latitude"]  -  df_test["pickup_latitude"]
    df_test["longitude_distance"] = df_test["dropoff_longitude"] - df_test["pickup_longitude"]   
    return df_test


def add_cluster_pair_avg_duration(df_test,imputer):
    gby = pd.read_csv(r'average trip duration by cluster pairs\cluster_pair_avg_duration.csv')
    kmeans = joblib.load(r'average trip duration by cluster pairs\kmeans.pkl')
    # Predict clusters
    df_test.loc[:,'pickup_cluster'] = kmeans.predict(df_test[['pickup_latitude','pickup_longitude']].values)
    df_test.loc[:,'dropoff_cluster'] = kmeans.predict(df_test[['dropoff_latitude','dropoff_longitude']].values)
    
    # Map mean duration back as a feature 
    df_test = df_test.merge(
        gby,
        on=['pickup_cluster','dropoff_cluster'],
        how='left'
    )
    
    imputed_test = imputer.transform(df_test)
    df_test = pd.DataFrame(imputed_test,columns=df_test.columns,index=df_test.index)

    return df_test


def split_data(df_test):
    x_test = df_test.drop(columns=['trip_duration'])
    t_test = df_test['trip_duration']
    return x_test,t_test




def encode(x_test,encoder,encoded_cols):
    # transform
    encoded_test = encoder.transform(x_test[encoded_cols])
    # convert to dataframe
    encoded_train_df = pd.DataFrame(
        encoded_test,
        columns=encoder.get_feature_names_out(encoded_cols),
        index=x_test.index
    )
    # drop original columns
    x_test.drop(columns=encoded_cols,inplace=True)
    # add encoded columns
    x_test = pd.concat([x_test,encoded_train_df],axis=1)

    return x_test


def evaluate( x_test , t_test , model,save_predicted = False):
    pred_t = model.predict(x_test)
    rmse =np.sqrt( mean_squared_error(t_test,pred_t) )
    r2 = r2_score(t_test,pred_t)
    print(f'rmse is {rmse} -- and r2_score is {r2}')
    if(save_predicted):
        pred_vals = pd.DataFrame({
            'Predicted_values': np.expm1(pred_t),
            'Real_values': np.expm1(t_test)
        })
        os.makedirs('results',exist_ok=True)
        pred_vals.to_csv(r'results\pred_vals.csv',index=False)

def prepare_data(args,model_dict):
    # upload Test Set
    df_test = pd.read_csv(test_args.test_path)
    # delete id and store_and_fwd_flag columns
    df_test.drop(columns=['id','store_and_fwd_flag'],inplace=True)
    # filter the data
    df_test = filter_data(df_test)
    # prepare date columns
    df_test = prepare_date_columns(df_test)
    # add rush_hour feature
    if(args.rush_hour):
        df_test = add_rush_hour(df_test)
    # add is_weekend feature
    if(args.is_weekend):
        df_test = add_is_weekend(df_test)
    # add distance features
    if(args.long_lat_distance):
        df_test = add_long_lat_distance(df_test)
    
    df_test['haversine_distance'] = haversine(
    df_test['pickup_latitude'].values,
    df_test['pickup_longitude'].values,
    df_test['dropoff_latitude'].values,
    df_test['dropoff_longitude'].values)

    # add cluster_pair_avg_duration
    if(args.cluster_pair_avg_duration):
        df_test = add_cluster_pair_avg_duration(df_test,model_dict['imputer'])
    
    # split data
    x_test,t_test = split_data(df_test)
    
    # encode
    if(args.encode):
        x_test = encode(x_test,model_dict['encoder'],args.encode)
    # scaling
    if(args.scaler):
        x_test = model_dict['scaler'].transform(x_test)
    # log target
    if(args.log_target):
        t_test = np.log1p(t_test)
        
    return x_test,t_test









if __name__ == '__main__':
    # Test Args
    parser = argparse.ArgumentParser(description='Testing The  Model')
    parser.add_argument("--test_path", type=str, default=r"split\test.csv",
                        help="Path to testing CSV file.")
    parser.add_argument("--model_utiliteis", type=str, default=r"model_utilities",
                        help="Path to the model dictionary")
    test_args = parser.parse_args()
    # Upload Args and Model
    with open (os.path.join(test_args.model_utiliteis,'args.json'),'r') as f:
        args = json.load(f)
        args = argparse.Namespace(**args)
    model_dict = joblib.load(os.path.join(test_args.model_utiliteis,'model_dict.pkl'))

    x_test,t_test = prepare_data(args,model_dict)
    evaluate(x_test,t_test,model_dict['model'],save_predicted=True)

    

    
