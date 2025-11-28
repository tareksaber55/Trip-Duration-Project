
import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures,StandardScaler
def filter_data(df_train,df_val):
    # handle trips out of New York
    min_lat = 40.49
    max_lat = 40.92
    min_lon = -74.27
    max_lon = -73.68
    df_train = df_train[(df_train['pickup_latitude'].between(min_lat,max_lat))& 
                        (df_train['pickup_longitude'].between(min_lon,max_lon))&
                        (df_train['dropoff_latitude'].between(min_lat,max_lat))& 
                        (df_train['dropoff_longitude'].between(min_lon,max_lon))]
    df_val = df_val[(df_val['pickup_latitude'].between(min_lat,max_lat))& 
                        (df_val['pickup_longitude'].between(min_lon,max_lon))&
                        (df_val['dropoff_latitude'].between(min_lat,max_lat))& 
                        (df_val['dropoff_longitude'].between(min_lon,max_lon))]
    
    # delete passenger_count values 0 and 7
    upper , lower = 7 , 0
    df_train = df_train[(df_train['passenger_count'] > lower) & (df_train['passenger_count'] < upper)]
    df_val = df_val[(df_val['passenger_count'] > lower) & (df_val['passenger_count'] < upper)]
    return df_train,df_val


def del_outliers(df_train,factor):
    Q1 = df_train['trip_duration'].quantile(0.25)
    Q3 = df_train['trip_duration'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    filtered_df = df_train[(df_train['trip_duration'] >= lower ) & (df_train['trip_duration'] <= upper )] 
    return filtered_df


def haversine(lat1, lon1, lat2, lon2):
    # convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Earth radius in km


def encode(x_train,x_val,encoded_cols):
    encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    encoder.fit(x_train[encoded_cols])
    # transform
    encoded_train = encoder.transform(x_train[encoded_cols])
    encoded_val = encoder.transform(x_val[encoded_cols])
    # convert to dataframe
    encoded_train_df = pd.DataFrame(
        encoded_train,
        columns=encoder.get_feature_names_out(encoded_cols),
        index=x_train.index
    )
    encoded_val_df = pd.DataFrame(
        encoded_val,
        columns=encoder.get_feature_names_out(encoded_cols),
        index=x_val.index
    )
    # drop original columns
    x_train.drop(columns=encoded_cols,inplace=True)
    x_val.drop(columns=encoded_cols,inplace=True)
    # add encoded columns
    x_train = pd.concat([x_train,encoded_train_df],axis=1)
    x_val = pd.concat([x_val,encoded_val_df],axis=1)

    return x_train,x_val,encoder




def split_data(df_train,df_val):
    x_train = df_train.drop(columns=['trip_duration'])
    x_val = df_val.drop(columns=['trip_duration'])
    t_train = df_train['trip_duration']
    t_val = df_val['trip_duration']
    return x_train,x_val,t_train,t_val

def prepare_date_columns(df_train,df_val):
    df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
    df_val['pickup_datetime'] = pd.to_datetime(df_val['pickup_datetime'])

    df_train['pickup_month'] = df_train['pickup_datetime'].dt.month
    df_val['pickup_month'] = df_val['pickup_datetime'].dt.month

    df_train['pickup_hour'] = df_train['pickup_datetime'].dt.hour
    df_val['pickup_hour'] = df_val['pickup_datetime'].dt.hour

    df_train['pickup_weekday'] = df_train['pickup_datetime'].dt.weekday
    df_val['pickup_weekday'] = df_val['pickup_datetime'].dt.weekday
    
    # delete date column
    df_train = df_train.drop(columns=['pickup_datetime'])
    df_val = df_val.drop(columns=['pickup_datetime'])

    return df_train,df_val

def add_rush_hour(df_train,df_val):
    df_train['rush_hour'] = df_train['pickup_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
    df_val['rush_hour'] = df_val['pickup_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
    return df_train,df_val

def add_is_weekend(df_train,df_val):
    df_train['is_weekend'] = df_train['pickup_weekday'].isin([5,6]).astype(int) # saturday and sunday
    df_val['is_weekend'] = df_val['pickup_weekday'].isin([5,6]).astype(int)
    return df_train,df_val

def add_long_lat_distance(df_train,df_val):
    df_train["latitude_distance"]  = df_train["dropoff_latitude"]  -  df_train["pickup_latitude"]
    df_train["longitude_distance"] = df_train["dropoff_longitude"] - df_train["pickup_longitude"]
    df_val["latitude_distance"]  = df_val["dropoff_latitude"]  - df_val["pickup_latitude"]
    df_val["longitude_distance"] = df_val["dropoff_longitude"] - df_val["pickup_longitude"]
    return df_train,df_val

def add_cluster_pair_avg_duration(df_train,df_val,imputer_name):
    gby = pd.read_csv(r'average trip duration by cluster pairs\cluster_pair_avg_duration.csv')
    kmeans = joblib.load(r'average trip duration by cluster pairs\kmeans.pkl')
    # Predict clusters
    df_train.loc[:,'pickup_cluster'] = kmeans.predict(df_train[['pickup_latitude','pickup_longitude']].values)
    df_train.loc[:,'dropoff_cluster'] = kmeans.predict(df_train[['dropoff_latitude','dropoff_longitude']].values)

    df_val.loc[:,'pickup_cluster'] = kmeans.predict(df_val[['pickup_latitude','pickup_longitude']].values)
    df_val.loc[:,'dropoff_cluster'] = kmeans.predict(df_val[['dropoff_latitude','dropoff_longitude']].values)
    # Map mean duration back as a feature 
    df_train = df_train.merge(
        gby,
        on=['pickup_cluster','dropoff_cluster'],
        how='left'
    )
    df_val = df_val.merge(
        gby,
        on=['pickup_cluster','dropoff_cluster'],
        how='left'
    )
    imputer = None
    if(imputer_name == 'knn'):
        imputer = KNNImputer().fit(df_train)           
    elif(imputer_name == 'simple'):
        imputer = SimpleImputer().fit(df_train)
    else:
        imputer = IterativeImputer(random_state=42).fit(df_train)
    imputed_train = imputer.transform(df_train)
    imputed_val = imputer.transform(df_val)
    df_train = pd.DataFrame(imputed_train,columns=df_train.columns,index=df_train.index)
    df_val = pd.DataFrame(imputed_val,columns=df_val.columns,index=df_val.index)

    return df_train,df_val,imputer


def scaling(x_train,x_val,scaler_name):
    
    if(scaler_name == 'standardscaler'):
        scaler = StandardScaler()
    elif(scaler_name == 'minmaxscaler'):
        scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_train = pd.DataFrame(x_train_scaled,columns=x_train.columns,index=x_train.index)
    x_val = pd.DataFrame(x_val_scaled,columns=x_val.columns,index=x_val.index)
    return x_train,x_val,scaler
        


