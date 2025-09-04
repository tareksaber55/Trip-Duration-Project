import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import haversine_distances


def prepare_date_cols(df_test):
    df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
    df_test['pickup_month'] = df_test['pickup_datetime'].dt.month
    df_test['pickup_hour'] = df_test['pickup_datetime'].dt.hour
    df_test['pickup_weekday'] = df_test['pickup_datetime'].dt.weekday
    df_test['is_weekend'] = df_test['pickup_weekday'].isin([5,6]).astype(int) # saturday and sunday
    df_test['jam_hour'] = (df_test['pickup_hour'].between(11,17) ).astype(int)
    return df_test



def haversine_distance(row):
    # Convert pickup and dropoff to radians
    start = np.radians([row['pickup_latitude'], row['pickup_longitude']])
    end   = np.radians([row['dropoff_latitude'], row['dropoff_longitude']])
    
    # sklearn expects 2D arrays → reshape
    result = haversine_distances([start, end])
    
    # distance in km (Earth radius = 6371 km)
    return result[0,1] * 6371



def prepare_cat_cols(df_test,encoder):
    

    cat_test = pd.DataFrame(encoder.transform(df_test[['passenger_count','store_and_fwd_flag','vendor_id','pickup_month','pickup_hour', 'pickup_weekday']]),
                                columns=encoder.get_feature_names_out(['passenger_count','store_and_fwd_flag','vendor_id','pickup_month','pickup_hour', 'pickup_weekday']),
                                index=df_test.index)

    df_test = pd.concat([df_test, cat_test], axis=1)
    return df_test



def scale_distribute(df_test,scaler):
    df_test = df_test.to_numpy()
    x_test = df_test[:,:-1]
    t_test = df_test[:,-1]
    x_test = scaler.transform(x_test)
    return x_test,t_test
    



def evaluate(x_test,t_test,model):
    pred_t = model.predict(x_test)
    rmse =np.sqrt( mean_squared_error(t_test,pred_t) )
    r2 = r2_score(t_test,pred_t)
    print(f'rmse is {rmse} -- and r2_score is {r2}')



def prepare_data(df_test  , encoder , scaler):
    df_test = prepare_date_cols(df_test)
    df_test['haversine_distance'] = df_test.apply(haversine_distance, axis=1)
    df_test = prepare_cat_cols(df_test,encoder)
    df_test['log_trip_duration'] = np.log1p(df_test['trip_duration'])
    features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'is_weekend', 'jam_hour', 'haversine_distance','passenger_count_1',
       'passenger_count_2', 'passenger_count_3', 'passenger_count_4',
       'passenger_count_5', 'passenger_count_6', 'store_and_fwd_flag_N',
       'store_and_fwd_flag_Y', 'vendor_id_1', 'vendor_id_2', 'pickup_month_1',
       'pickup_month_2', 'pickup_month_3', 'pickup_month_4', 'pickup_month_5',
       'pickup_month_6', 'pickup_hour_0', 'pickup_hour_1', 'pickup_hour_2',
       'pickup_hour_3', 'pickup_hour_4', 'pickup_hour_5', 'pickup_hour_6',
       'pickup_hour_7', 'pickup_hour_8', 'pickup_hour_9', 'pickup_hour_10',
       'pickup_hour_11', 'pickup_hour_12', 'pickup_hour_13', 'pickup_hour_14',
       'pickup_hour_15', 'pickup_hour_16', 'pickup_hour_17', 'pickup_hour_18',
       'pickup_hour_19', 'pickup_hour_20', 'pickup_hour_21', 'pickup_hour_22',
       'pickup_hour_23', 'pickup_weekday_0', 'pickup_weekday_1',
       'pickup_weekday_2', 'pickup_weekday_3', 'pickup_weekday_4',
       'pickup_weekday_5', 'pickup_weekday_6', 'log_trip_duration' ]
    df_test = df_test[features]
    x_test,t_test = scale_distribute(df_test,scaler)
    return x_test,t_test


if __name__ == '__main__':
    df_test = pd.read_csv(r'split\test.csv')
    model = joblib.load("model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    x_test,t_test = prepare_data(df_test,encoder,scaler)
    evaluate(x_test,t_test,model)
    
