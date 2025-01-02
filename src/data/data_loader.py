import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import logging
from src.utils.utils import validate_dataset

logger = logging.getLogger(__name__)

class CTUDataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.features = config['features']
        self.target = config['target']
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate dataset structure and contents"""
        required_columns = self.features + [self.target]
        return validate_dataset(df, required_columns)
    
    def preprocess_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial preprocessing of raw network data"""
        # Handle missing values
        df = df.fillna(0)
        
        # Convert timestamps to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Normalize IP addresses
        if 'src_ip' in df.columns:
            df['src_ip'] = df['src_ip'].apply(lambda x: int(ip2long(x)))
        
        return df
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare CTU-13 dataset"""
        try:
            df = pd.read_csv(data_path)
            
            if not self.validate_data(df):
                raise ValueError("Dataset validation failed")
                
            df = self.preprocess_raw_data(df)
            
            X = df[self.features]
            y = df[self.target]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise