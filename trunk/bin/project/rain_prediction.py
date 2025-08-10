import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(r'weather_dataset.csv')
    return df

# Preprocess the hourly data
def preprocess_data(df):
    # Convert Formatted Date to datetime and handle timezone
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df['Date'] = df['Formatted Date'].dt.date
    df['Hour'] = df['Formatted Date'].dt.hour
    
    # Aggregate hourly data to daily data
    daily_df = df.groupby('Date').agg({
        'Temperature (C)': ['mean', 'min', 'max'],
        'Apparent Temperature (C)': ['mean', 'min', 'max'],
        'Humidity': ['mean', 'max'],
        'Wind Speed (km/h)': ['mean', 'max'],
        'Wind Bearing (degrees)': 'mean',
        'Visibility (km)': 'mean',
        'Pressure (millibars)': ['mean', 'min', 'max'],
        'Precip Type': lambda x: 1 if 'rain' in x.values else 0  # 1 if any hour has rain
    }).reset_index()
    
    # Flatten column names
    daily_df.columns = [
        'Date',
        'Temp_Mean', 'Temp_Min', 'Temp_Max',
        'Apparent_Temp_Mean', 'Apparent_Temp_Min', 'Apparent_Temp_Max',
        'Humidity_Mean', 'Humidity_Max',
        'Wind_Speed_Mean', 'Wind_Speed_Max',
        'Wind_Bearing_Mean',
        'Visibility_Mean',
        'Pressure_Mean', 'Pressure_Min', 'Pressure_Max',
        'Rain_Today'
    ]
    
    # Create target variable: RainTomorrow (shift Rain_Today by 1 day)
    daily_df['RainTomorrow'] = daily_df['Rain_Today'].shift(-1)
    
    # Drop last row (no RainTomorrow value) and rows with missing values
    daily_df = daily_df.dropna()
    
    # Extract additional date features
    daily_df['DayOfYear'] = pd.to_datetime(daily_df['Date']).dt.dayofyear
    daily_df['Month'] = pd.to_datetime(daily_df['Date']).dt.month
    
    # Select features for the model
    features = [
        'Temp_Mean', 'Temp_Min', 'Temp_Max',
        'Apparent_Temp_Mean', 'Apparent_Temp_Min', 'Apparent_Temp_Max',
        'Humidity_Mean', 'Humidity_Max',
        'Wind_Speed_Mean', 'Wind_Speed_Max',
        'Wind_Bearing_Mean',
        'Visibility_Mean',
        'Pressure_Mean', 'Pressure_Min', 'Pressure_Max',
        'DayOfYear', 'Month'
    ]
    
    X = daily_df[features]
    y = daily_df['RainTomorrow'].astype(int)
    
    return X, y, daily_df

# Feature scaling
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Train and evaluate model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return rf_model, X_test, y_test, y_pred, metrics, cv_scores

# Visualize feature importance
def plot_feature_importance(model, features):
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': features, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance in Random Forest Model')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    # Load data (update path to your local dataset)
    file_path = "weather_data.csv"  
    df = load_data(file_path)
    
    # Preprocess data
    X, y, daily_df = preprocess_data(df)
    
    # Scale features
    X_scaled, scaler = scale_features(X)
    
    # Train and evaluate model
    model, X_test, y_test, y_pred, metrics, cv_scores = train_model(X_scaled, y)
    
    # Print results
    print("Model Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Plot feature importance
    plot_feature_importance(model, X.columns)
    
    print("\nFeature importance plot and confusion matrix saved as PNG files.")