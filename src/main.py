import logging
import argparse
from pathlib import Path
from src.utils.config import load_config, DEFAULT_CONFIG
from src.data.data_loader import CTUDataLoader
from src.features.feature_engineering import NetworkFeatureEngineer
from src.models.model import BotnetDetectionModel
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Botnet Detection System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to input data')
    return parser.parse_args()

def main():
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config) if Path(args.config).exists() else DEFAULT_CONFIG
    
    try:
        # Initialize components
        data_loader = CTUDataLoader(config['data'])
        feature_engineer = NetworkFeatureEngineer(config['features'])
        model = BotnetDetectionModel(config['model'])
        trainer = ModelTrainer(config['training'])
        evaluator = ModelEvaluator(config['evaluation'])
        
        # Load and preprocess data
        logger.info("Loading data...")
        X, y = data_loader.load_data(args.data)
        
        # Engineer features
        logger.info("Engineering features...")
        X_processed = feature_engineer.engineer_features(X)
        
        # Train and evaluate model
        logger.info("Training model...")
        trained_model = trainer.train(model, X_processed, y)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluator.evaluate_model(trained_model, X_processed, y)
        
        # Perform cross-validation
        logger.info("Performing cross-validation...")
        cv_scores = evaluator.cross_validate(trained_model, X_processed, y)
        
        logger.info("Botnet detection pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
