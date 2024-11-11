# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import os

# def load_and_clean_data(data_path):
#     """
#     Loads and cleans data from the provided path and saves the processed data.
    
#     Args:
#         data_path (str): The path to the raw data file (CSV).
    
#     Returns:
#         pd.DataFrame: The cleaned data.
#     """
#     # Load the raw data (assumed to be CSV for this example)
#     data = pd.read_csv(data_path)

#     # Check for missing values and handle them (drop or impute)
#     print("Checking for missing values...")
#     print(data.isnull().sum())

#     # Drop rows with missing values (can be changed to imputation if needed)
#     data = data.dropna()  # Drop rows with missing values
#     print(f"Data shape after dropping missing values: {data.shape}")

#     # Convert 'date' to datetime and extract features like year, month, and day
#     data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
#     data['year'] = data['date'].dt.year
#     data['month'] = data['date'].dt.month
#     data['day'] = data['date'].dt.day
#     data = data.drop(columns=['date'])  # Drop the original date column

#     # Handle categorical columns (encoding)
#     categorical_columns = data.select_dtypes(include=['object']).columns
#     label_encoder = LabelEncoder()

#     for col in categorical_columns:
#         print(f"Encoding column: {col}")
#         data[col] = label_encoder.fit_transform(data[col])

#     # Scale numerical columns (e.g., features in ML models)
#     numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
#     scaler = StandardScaler()
    
#     # Scale numerical columns
#     data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
#     print("Data after scaling numerical columns:\n", data.head())

#     # Ensure the 'processed' directory exists
#     processed_data_path = "./data/processed"
#     os.makedirs(processed_data_path, exist_ok=True)  # Create the directory if it doesn't exist
    
#     # Define the path to save the cleaned data
#     cleaned_data_path = os.path.join(processed_data_path, 'cleaned_data.csv')

#     # Save the cleaned data
#     data.to_csv(cleaned_data_path, index=False)
#     return data

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_and_clean_data(data_path):
    """
    Loads and cleans data from the provided path and saves the processed data.
    
    Args:
        data_path (str): The path to the raw data file (CSV).
    
    Returns:
        pd.DataFrame: The cleaned data.
    """
    # Load the raw data
    data = pd.read_csv(data_path)

    # Drop rows with missing values (adjust if imputation is preferred)
    data = data.dropna()

    # Convert 'date' to datetime and extract year, month, and day
    data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data = data.drop(columns=['date'])

    # Encode categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Scale numerical columns, excluding the target column 'disease'
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.difference(['disease'])
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Ensure 'processed' directory exists
    processed_data_path = "./data/processed"
    os.makedirs(processed_data_path, exist_ok=True)
    cleaned_data_path = os.path.join(processed_data_path, 'cleaned_data.csv')
    data.to_csv(cleaned_data_path, index=False)

    return data
