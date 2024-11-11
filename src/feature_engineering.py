import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_features(data):
    """
    Extract features from the data, scaling numerical columns.
    
    Args:
        data (pd.DataFrame): The cleaned data.
    
    Returns:
        pd.DataFrame: Scaled numerical data.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.select_dtypes(include=["float64", "int64"]))
    scaled_data = pd.DataFrame(data_scaled, columns=data.select_dtypes(include=["float64", "int64"]).columns)
    return scaled_data
