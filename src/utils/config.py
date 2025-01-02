import yaml
import os
from pathlib import Path
from typing import Dict

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return DEFAULT_CONFIG
    except Exception as e:
        raise Exception(f"Error loading config: {str(e)}")

# Default configuration settings
DEFAULT_CONFIG = {
    'data': {
        'features': [
            'duration',
            'protocol_type',
            'service',
            'flag',
            'src_bytes',
            'dst_bytes',
            'land',
            'wrong_fragment',
            'urgent',
            'src_ip',
            'dst_ip',
            'timestamp',
            'packet_count',
            'byte_count',
            'flow_id',
            'payload_bytes',
            'tcp_flags'
        ],
        'target': 'label',
        'test_size': 0.2,
        'random_state': 42
    },
    'features': {
        'scaling': {
            'duration': 'standard',
            'src_bytes': 'standard',
            'dst_bytes': 'standard',
            'packet_count': 'standard',
            'byte_count': 'standard'
        },
        'categorical_features': [
            'protocol_type',
            'service',
            'flag'
        ]
    },
    'model': {
        'model_type': 'rf',
        'rf_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },
        'gbm_params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3
        },
        'svm_params': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        },
        'dnn_params': {
            'input_dim': None,  # Will be set dynamically
            'batch_size': 32,
            'epochs': 50,
            'validation_split': 0.2
        },
        'lstm_params': {
            'sequence_length': 10,
            'input_dim': None,  # Will be set dynamically
            'batch_size': 32,
            'epochs': 50
        }
    },
    'training': {
        'cross_validation': {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 42
        },
        'model_path': 'models/botnet_model.pkl'
    },
    'evaluation': {
        'metrics': [
            'accuracy',
            'precision',
            'recall',
            'f1',
            'auc_roc'
        ],
        'threshold': 0.5
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/botnet_detection.log'
    }
}