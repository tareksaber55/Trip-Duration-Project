
# Trip-Duration-Project 🚖
Overview

Trip-Duration-Project is a complete machine-learning pipeline designed to predict taxi trip duration using trip metadata such as pickup/dropoff coordinates, timestamps, passenger count, and engineered features.
The project includes data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and saving prediction outputs.

Repository Structure
├── weather data/              # weather or external data for enrichment  
├── split/                     # scripts for train/valid/test splitting
├── results/                   # prediction outputs, metrics  
├── model_utilities/           # helper functions for modeling, saving, loading  
├── feature_engineering.py     # script that creates new features  
├── perpare_kmeans.py          # k-means clustering for pickup/dropoff coordinates  
├── main.py                    # main ML pipeline to train model  
├── test.py                    # script to test/evaluate the model  
├── trip-duration EDA.ipynb    # notebook for data exploration & visualization  
└── README.md                  # this file  

Data & Features

The project expects typical NYC taxi data fields such as:

pickup_datetime

dropoff_datetime

pickup_latitude, pickup_longitude

dropoff_latitude, dropoff_longitude

passenger_count

vendor_id

From this raw data, the feature engineering step produces additional features such as:

Time-based features

pickup hour

pickup month

day of week

is_weekend

rush_hour

Distance & geospatial features

longitude distance

latitude distance

haversine distance

k-means cluster features

cluster pair average duration

Other engineered features

log-target transformation (optional)

outlier removal

one-hot encoding for categorical variables

Model Training Pipeline
1. Preprocessing & Feature Engineering

Run feature_engineering.py to transform raw data into numerical features.

2. Clustering (optional)

Run perpare_kmeans.py to learn pickup/dropoff location clusters.

3. Train/Validation/Test Split

Scripts inside split/ generate correct datasets.

4. Training

Run main.py to train the model.
Models may include: RandomForest, XGBoost, Gradient Boosting, etc.

5. Evaluation

Run test.py to generate predictions and compute metrics such as:

RMSE

R²

All results will be saved inside the results/ directory.

6. EDA

The notebook trip-duration EDA.ipynb allows visual analysis of distributions, correlations, and feature importance.

Example Usage
# clone the project
git clone https://github.com/tareksaber55/Trip-Duration-Project
cd Trip-Duration-Project

# Extract The Split file

# install requirements (if requirements.txt exists)
pip install -r requirements.txt

# run feature engineering
python feature_engineering.py

# optional clustering
python perpare_kmeans.py

# train the model (add args)
python main.py 

# evaluate on test set
python test.py

Results

The results/ folder stores:

Predicted values

EDA / Modeling Summary

  
Saved models


Future Improvements

Possible enhancements:

Hyperparameter tuning (GridSearch, RandomizedSearch)

Adding  traffic data

Using advanced models (XGBoost, LightGBM, deep learning)

</pre>
