from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import logging
from ..visualization.plots import plot_confusion_matrix, plot_roc_curve

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.metrics = {}
        
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation"""
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            self.metrics['accuracy'] = accuracy_score(y_test, y_pred)
            self.metrics['precision'] = precision_score(y_test, y_pred)
            self.metrics['recall'] = recall_score(y_test, y_pred)
            self.metrics['f1'] = f1_score(y_test, y_pred)
            self.metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Generate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Generate classification report
            class_report = classification_report(y_test, y_pred)
            
            # Create visualizations
            plot_confusion_matrix(conf_matrix, ['Normal', 'Botnet'])
            plot_roc_curve(y_test, y_pred_proba)
            
            logger.info(f"Model Evaluation Results:")
            logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
            logger.info(f"Precision: {self.metrics['precision']:.4f}")
            logger.info(f"Recall: {self.metrics['recall']:.4f}")
            logger.info(f"F1 Score: {self.metrics['f1']:.4f}")
            logger.info(f"AUC-ROC: {self.metrics['auc_roc']:.4f}")
            logger.info("\nClassification Report:\n", class_report)
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
            
    def cross_validate(self, model, X, y, n_splits=5):
        """Perform cross-validation"""
        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            cv_scores = {
                'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
                'precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
                'recall': cross_val_score(model, X, y, cv=cv, scoring='recall'),
                'f1': cross_val_score(model, X, y, cv=cv, scoring='f1')
            }
            
            for metric, scores in cv_scores.items():
                logger.info(f"Cross-validation {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            return cv_scores
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise