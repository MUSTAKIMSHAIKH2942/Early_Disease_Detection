import matplotlib.pyplot as plt

def plot_feature_importances(model, feature_names):
    """
    Plots the feature importances of the trained model.
    
    Args:
        model: The trained machine learning model.
        feature_names (list): List of feature names.
    """
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance')
    plt.barh(range(len(feature_names)), importances[indices], align="center")
    plt.yticks(range(len(feature_names)), feature_names[indices])
    plt.xlabel('Relative Importance')
    plt.savefig("./images/feature_importance.jpg")
