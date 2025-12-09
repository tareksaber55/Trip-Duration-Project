
# Trip-Duration-Project 🚖

## Overview

Trip-Duration-Project is a machine-learning pipeline to predict taxi trip duration based on trip metadata (pickup & dropoff coordinates, timestamps, and optional additional features).  
The project includes data preprocessing / feature engineering, exploratory data analysis (EDA), model training & evaluation, and outputs for predictions and performance metrics.

## Repository Structure
<pre>
├── weather data/ # (optional) weather or external data used for enrichment
├── average trip duration by cluster pairs/ #contain average trip duration between every pair of clusters csv file and kmeans model to predict pickup_cluster and dropoff_cluster 
├── split/ #(train/validation/test) data
├── results/ # evaluation results, prediction outputs, model performance reports
├── model_utilities/ # saved model ,args and metrics
├── feature_engineering.py # script to create features (distance, clusters, time-based, etc.)
├── perpare_kmeans.py # script to cluster pickup/dropoff locations (if clustering used)
├── main.py # main script to run full training pipeline
├── test.py # script to run inference / evaluation on test set
├── trip-duration EDA.ipynb # notebook for exploratory data analysis & visualization
└── README.md # this file
</pre>

## Data & Features

The expected input data includes (but is not limited to) the following fields (similar to standard taxi-trip datasets):  
- `pickup_datetime` — timestamps to start trip
- `pickup_latitude`, `pickup_longitude`, `dropoff_latitude`, `dropoff_longitude` — geospatial coordinates of start and end 
- `passenger_count`, `vendor_id`, `store_and_fwd_flag`

From these, the feature-engineering script derives additional features, e.g.:  
- Geospatial distance (e.g. Haversine) between pickup & dropoff  
- Cluster-based zone features like average trip duration between every pair of clusters , pickup_cluster , dropoff_cluster (using K-means clustering of locations)  
- Temporal features: hour of day, day of week, month, rush-hour flags, is_weekend.  
- (Optional) External features, e.g. weather data — depending on `weather data/` usage.  

## Model Training & Evaluation Pipeline

Use the following high-level workflow:

1. **perpare_kmeans.py**  
   Run `perpare_kmeans.py` to prepare average_trip_duration_between_every_pair_of_clusters file and kmeans model   
   You can choose number of clusters of kmeans model.
   
3. **Train / validation split**  
   Use code under `split/` to partition data properly (train / validation / test).  

4. **Training**  
   Run `main.py` to train Ridge Model(you can add more to the code) — you can choose The Features you need, and save the best-performing model using utilities under `model_utilities/`.  

5. **Evaluation & testing**  
   Run `test.py` (or equivalent) on held-out test data to assess performance (RMSE, R²). Results and prediction outputs will be saved under `results/`.  

6. **Analysis & visualization**  
   Use `trip-duration EDA.ipynb` to explore data distributions, feature importance, residual analysis, outliers, and to guide feature engineering or model improvements.  

## Example Usage

```bash
# clone the repository
git clone https://github.com/tareksaber55/Trip-Duration-Project.git
cd Trip-Duration-Project

# (optional) install dependencies if you have requirements.txt
pip install -r requirements.txt


# run clustering if using k-means
python perpare_kmeans.py

# run main training pipeline
python main.py --del_outliers --log_target --cluster_pair_avg_duration --encode passenger_count  pickup_month  pickup_weekday pickup_hour  vendor_id --scaler standardscaler

# evaluate / test model
python test.py

Performance metrics on validation / test set (RMSE, R²)
```
## Results on Test Set

•	RMSE: 0.4980

•	R² Score: 0.6066

•  **Note**: You can get more better results if you use more advanced models, but in this project only ridge model with alpha 1 is permitted to make the challenge  harder

## Possible Extensions & Future Work

Some ideas to further improve or extend the project:

Hyperparameter tuning (e.g. grid search / randomized search ) to improve model performance.

Incorporating external data:  traffic, city event schedules — to capture real-world conditions.

Building a real-time inferencing API or web service that accepts pickup/dropoff coordinates and time, returns predicted duration.

More advanced modeling: ensemble methods, gradient boosting, or deep learning approaches for spatio-temporal modeling.

