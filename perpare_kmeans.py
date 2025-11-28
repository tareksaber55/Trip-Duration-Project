import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

def del_outliers(df_train):
    Q1 = df_train['trip_duration'].quantile(0.25)
    Q3 = df_train['trip_duration'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    filtered_df = df_train[(df_train['trip_duration'] >= lower ) & (df_train['trip_duration'] <= upper )] 
    return filtered_df

if __name__ == '__main__':
    df_train = pd.read_csv(r'split\train.csv')
    df_train = del_outliers(df_train)
    # train k_means -> prepare_avg_per_clusters
    coords = np.vstack([
    df_train[['pickup_latitude', 'pickup_longitude']].values,
    df_train[['dropoff_latitude', 'dropoff_longitude']].values
    ])

    kmeans = KMeans(n_clusters=100, random_state=42).fit(coords)

    # Predict clusters
    df_train.loc[:,'pickup_cluster'] = kmeans.predict(df_train[['pickup_latitude','pickup_longitude']].values)
    df_train.loc[:,'dropoff_cluster'] = kmeans.predict(df_train[['dropoff_latitude','dropoff_longitude']].values)



    # Compute mean trip duration per pair 
    gby = df_train.groupby(['pickup_cluster','dropoff_cluster'])['trip_duration'].mean() 
    gby.name = 'avg_duration_by_clusterpair'

    # Map mean duration back as a feature 
    df_train = df_train.merge(
        gby,
        on=['pickup_cluster','dropoff_cluster'],
        how='left'
    ) 

    print(df_train.head())
    dir_name = 'average trip duration by cluster pairs'
    os.makedirs(dir_name,exist_ok=True)
    joblib.dump(kmeans,os.path.join(dir_name,'kmeans.pkl'))
    gby.to_frame().reset_index().to_csv(os.path.join(dir_name,'cluster_pair_avg_duration.csv'), index=False)
