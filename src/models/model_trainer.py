from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)

class BotnetModelTrainer:
    def __init__(self, model_path: str = 'models/botnet_model.pkl'):
        self.model_path = model_path
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
    def train(self, X, y) -> Tuple[float, float]:
        """
        Train the botnet detection model
        """
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Calculate scores
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Save the model
            joblib.dump(self.model, self.model_path)
            
            logger.info(f"Model trained successfully. Test score: {test_score:.4f}")
            return train_score, test_score
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise