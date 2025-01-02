import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import logging

logger = logging.getLogger(__name__)

class BotnetDetectionModel:
    def __init__(self, config: dict):
        self.config = config
        self.model_type = config['model_type']
        self.models = {
            'rf': self._create_random_forest,
            'gbm': self._create_gradient_boosting,
            'svm': self._create_svm,
            'dnn': self._create_deep_neural_network,
            'lstm': self._create_lstm
        }
        
    def _create_random_forest(self):
        return RandomForestClassifier(
            n_estimators=self.config['rf_params']['n_estimators'],
            max_depth=self.config['rf_params']['max_depth'],
            random_state=42
        )
        
    def _create_gradient_boosting(self):
        return GradientBoostingClassifier(
            n_estimators=self.config['gbm_params']['n_estimators'],
            learning_rate=self.config['gbm_params']['learning_rate'],
            random_state=42
        )
        
    def _create_svm(self):
        return SVC(
            kernel=self.config['svm_params']['kernel'],
            C=self.config['svm_params']['C'],
            probability=True
        )
        
    def _create_deep_neural_network(self):
        model = Sequential([
            Dense(256, activation='relu', input_shape=(self.config['input_dim'],)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
        
    def _create_lstm(self):
        model = Sequential([
            LSTM(128, input_shape=(self.config['sequence_length'], self.config['input_dim'])),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
        
    def get_model(self):
        if self.model_type not in self.models:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        return self.models[self.model_type]()