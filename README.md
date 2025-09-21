🗽 NYC Taxi Trip Duration Prediction

This project predicts taxi trip duration in New York City using machine learning. The dataset comes from the Kaggle competition
[data link] https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview

📂 Project Structure

├── trip-duration EDA.ipynb   # Exploratory Data Analysis & feature engineering

├── main.py                   # Training script (model building & saving)

├── test.py                   # Evaluation script on unseen test data

⚙️ Workflow

1 - EDA & Feature Engineering (trip-duration EDA.ipynb)

-Convert datetime into month, hour, weekday.

-Add is_weekend, jam_hour (rush hour indicator).

-Compute geospatial features:

1-haversine_distance (pickup → dropoff straight-line distance).

2-bearing, euclidean_distance, manhattan_distance (optional).

3-pickup_cluster and drop cluster

4-average trip duration per every cluster pairs

-One-Hot Encode categorical features (passenger_count, vendor_id, etc.).

-Apply log transformation: log_trip_duration = log(1+trip_duration)


2 - Model Training (main.py)

-Prepares training & validation sets.

-Applies scaling (StandardScaler or MinMaxScaler).

-Trains regression models (Ridge, RandomForest, XGBoost).

-Saves the best model (model.pkl) along with preprocessing objects (encoder.pkl, scaler.pkl).

3 - Model Evaluation (test.py)

-Loads the trained model, encoder, and scaler.

-Applies the same feature engineering pipeline on new test data.

-Evaluates using RMSE and R² metrics on log_trip_duration.

🧮 Features Used

-Datetime features: pickup month, hour, weekday, weekend indicator, rush-hour flag.

-Geospatial features: pickup/dropoff coordinates, haversine distance , pickup & dropoff cluster , average trip duration per every cluster pairs.

-Categorical features (OHE): passenger count, vendor ID, store-and-forward flag, month, hour, weekday.

-Target variable: log_trip_duration.

📊 Metrics

-The project evaluates performance using:

-RMSE (Root Mean Squared Error)

-R² Score (Coefficient of Determination)

-Both computed on the log-transformed target .

🚀 How to Run

1 - Install requirements:

-pip install -r requirements.txt


(or manually: pandas numpy scikit-learn matplotlib seaborn xgboost joblib)

2 - Run training:

-python main.py


3 - Run evaluation on test set:

-python test.py

📌 Next Steps

-Hyperparameter tuning (RandomForest / XGBoost).
-Consider traffic/weather data for more accuracy.
