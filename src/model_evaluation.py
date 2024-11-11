from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

def evaluate_model(model, data, model_name, report_path="./reports/model_evaluation.csv", best_model_path="./models/best_model.pkl"):
    """
    Evaluates the model, saves the metrics to a report file, and highlights the best model.

    Args:
        model: Trained model.
        data (pd.DataFrame): The processed data.
        model_name (str): Name of the model.
        report_path (str): Path to save the evaluation report.
        best_model_path (str): Path to save the best model.
    """
    X = data.drop('disease', axis=1)
    y = data['disease']
    y_pred = model.predict(X)
    
    # Metrics calculation
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, average='weighted'),
        'Recall': recall_score(y, y_pred, average='weighted'),
        'F1 Score': f1_score(y, y_pred, average='weighted')
    }

    # Load previous metrics if the report file exists, otherwise create a new DataFrame
    if os.path.exists(report_path):
        report_df = pd.read_csv(report_path)
        # Use pd.concat to append the new metrics
        report_df = pd.concat([report_df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        report_df = pd.DataFrame([metrics])
    
    # Identify and highlight the best model based on F1 score
    best_model = report_df.loc[report_df['F1 Score'].idxmax()]
    report_df.to_csv(report_path, index=False)

    # Save the best model if it's the current one
    if metrics['F1 Score'] == best_model['F1 Score']:
        joblib.dump(model, best_model_path)
        print(f"Best model '{model_name}' saved with F1 Score: {best_model['F1 Score']:.2f}")

    # Plot confusion matrix
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(f"./images/{model_name}_confusion_matrix.png")
    plt.close()

    print(f"Metrics for {model_name} saved to {report_path}")
