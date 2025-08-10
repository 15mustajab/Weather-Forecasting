# Weather-Forecasting
# Rain Prediction Model Report
# Overview
This project develops a machine learning model for XYZ to predict next-day rain (binary classification: "Rain" or "No Rain") using historical hourly weather data. The model supports weather-sensitive industries like agriculture, logistics, and event planning by providing accurate short-term forecasts.
Dataset
The dataset contains hourly weather observations with features like Temperature (C), Humidity, Wind Speed (km/h), Pressure (millibars), and Precip Type. Data is aggregated to daily summaries (mean, min, max) for model training.
Methodology

Preprocessing: Aggregated hourly data to daily features, created RainTomorrow target by shifting daily rain indicator, and extracted date-based features (day of year, month).
Model: Random Forest Classifier (100 estimators, max depth 10) for robust handling of non-linear weather patterns.
Features: Temperature (mean/min/max), apparent temperature, humidity, wind speed, pressure, visibility, and date features.
Evaluation: Assessed using accuracy, precision, recall, F1 score, and 5-fold cross-validation.

Results

Performance: Achieved high accuracy (0.9739), precision (0.9818), recall (0.9908), and F1 score (0.0.9863)
Cross-validation scores: [0.97200622 0.97045101 0.96578538 0.97356143 0.97507788]
Average CV score: 0.9714 (+/- 0.0064)

Key Features: Humidity, pressure, and temperature were top predictors.
Visualizations: 
<img width="800" height="600" alt="confusion_matrix" src="https://github.com/user-attachments/assets/e6df6a15-3f74-4884-b4da-fe2e91f9ce43" />
<img width="1000" height="600" alt="feature_importance" src="https://github.com/user-attachments/assets/da62fb69-54fc-4f08-b44f-27bf5ee60e56" />


# Usage
Install dependencies: pip install pandas numpy scikit-learn matplotlib seaborn
Update file_path in rain_prediction.py to point to weather_data.csv.
Run the script to train the model and generate outputs.

