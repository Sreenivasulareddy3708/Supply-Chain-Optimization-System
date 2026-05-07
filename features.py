# features.py
import pandas as pd
import numpy as np

def create_features(df, base_forecast, date_col='date', demand_col='demand'):
    """
    Create feature matrix for XGBoost (Component 2 Input)
    Matches Appendix B.3
    
    Args:
        df: DataFrame containing historical data
        base_forecast: The trend forecast from PatchTST
    """
    features = df.copy()
    
    # 1. Incorporate PatchTST Forecast
    # Note: In a real scenario, you align the forecast with the timestamps
    features['patchtst_pred'] = base_forecast
    
    # Ensure date column is datetime
    if not np.issubdtype(features[date_col].dtype, np.datetime64):
        features[date_col] = pd.to_datetime(features[date_col])

    # 2. Temporal Features
    features['day_of_week'] = features[date_col].dt.dayofweek
    features['day_of_month'] = features[date_col].dt.day
    features['month'] = features[date_col].dt.month
    features['quarter'] = features[date_col].dt.quarter
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    
    # 3. Lag Features
    # Note: For forecasting horizon H, simple lags might not be available 
    # for all future steps (recursive strategy required), 
    # but based on the paper, we assume availability or use training mode logic.
    for lag in [1, 7, 14, 30]:
        features[f'lag_{lag}'] = features[demand_col].shift(lag)
    
    # 4. Rolling Statistics
    for window in [7, 30]:
        features[f'rolling_mean_{window}'] = features[demand_col].shift(1).rolling(window).mean()
        features[f'rolling_std_{window}'] = features[demand_col].shift(1).rolling(window).std()
        features[f'rolling_max_{window}'] = features[demand_col].shift(1).rolling(window).max()
        features[f'rolling_min_{window}'] = features[demand_col].shift(1).rolling(window).min()
    
    # 5. Differential Features
    features['lag_diff_1'] = features[demand_col].diff(1)
    features['lag_diff_7'] = features[demand_col].diff(7)
    
    # Drop NaNs created by lags
    features = features.dropna()
    
    # Separate X (features) and y (target) implied logic handled in training loop
    return features
