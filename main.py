import os
import yaml
import logging
from src.data_preprocessing import load_and_clean_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Loads the configuration file."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    try:
        # Load configuration
        config = load_config()

        # Data loading and preprocessing
        raw_data_path = config['data_paths']['raw_data']
        processed_data_path = config['data_paths']['processed_data']
        
        logger.info("Loading and cleaning data...")
        data = load_and_clean_data(raw_data_path)

        # Check if the processed data file already exists
        processed_file_path = os.path.join(processed_data_path, "cleaned_data.csv")
        if os.path.exists(processed_file_path):
            logger.info(f"File {processed_file_path} already exists. Loading the existing file.")
        else:
            os.makedirs(processed_data_path, exist_ok=True)
            data.to_csv(processed_file_path, index=False)

        # Model training
        logger.info("Training the model...")
        model_type = config['model_parameters']['model_type']
        model = train_model(data, model_type=model_type)

        # Model evaluation
        logger.info("Evaluating the model...")
        # Pass model_name as a string
        evaluate_model(model, data, model_name="RandomForest", report_path="./reports/model_evaluation.csv", best_model_path="./models/best_model.pkl")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
