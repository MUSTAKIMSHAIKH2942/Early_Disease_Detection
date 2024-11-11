# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# import xgboost as xgb
# from sklearn.model_selection import train_test_split

# def train_model(data, model_type='RandomForest'):
#     """
#     Trains a model based on the provided model_type (classification or regression).
#     Automatically selects classifier or regressor based on the nature of the target variable.
#     """
#     X = data.drop('disease', axis=1)  # Features
#     y = data['disease']  # Target

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # If the target is binary or multiclass classification (discrete labels), use a classifier
#     if y.nunique() <= 2 or model_type == 'RandomForest':  # Binary classification or user-specified classifier
#         if model_type == 'RandomForest':
#             model = RandomForestClassifier(n_estimators=100, max_depth=10)
#         elif model_type == 'XGBoost':
#             model = xgb.XGBClassifier(n_estimators=100, max_depth=10)
#         else:
#             raise ValueError(f"Unknown model type for classification: {model_type}")
#     # If the target is continuous (regression), use a regressor
#     else:
#         if model_type == 'RandomForest':
#             model = RandomForestRegressor(n_estimators=100, max_depth=10)
#         elif model_type == 'XGBoost':
#             model = xgb.XGBRegressor(n_estimators=100, max_depth=10)
#         else:
#             raise ValueError(f"Unknown model type for regression: {model_type}")

#     # Train the model
#     model.fit(X_train, y_train)
#     return model

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_model(data, model_type='RandomForest'):
    """
    Trains a model based on the provided model_type (classification or regression).
    """
    # Ensure 'disease' is treated as categorical
    data['disease'] = data['disease'].astype(int)
    
    X = data.drop('disease', axis=1)
    y = data['disease']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose the model based on model_type
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, max_depth=10)
    elif model_type == 'XGBoost':
        model = xgb.XGBClassifier(n_estimators=100, max_depth=10)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train the model
    model.fit(X_train, y_train)
    return model
