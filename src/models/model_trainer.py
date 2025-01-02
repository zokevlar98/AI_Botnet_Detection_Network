from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get('model_path', 'models/botnet_model.pkl')
        
    def train(self, model, X, y) -> Tuple[float, float]:
        """
        Train the botnet detection model
        """
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42)
            )
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Calculate scores
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Save the model
            joblib.dump(model, self.model_path)
            
            logger.info(f"Model trained successfully. Test score: {test_score:.4f}")
            return train_score, test_score
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise