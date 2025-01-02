import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple
import logging
from ..utils.utils import calculate_entropy

logger = logging.getLogger(__name__)

class NetworkFeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df['flow_duration'] = df['end_time'] - df['start_time']
        df['packets_per_second'] = df['packet_count'] / df['flow_duration']
        df['bytes_per_second'] = df['byte_count'] / df['flow_duration']
        return df
        
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical network features"""
        # Packet size statistics
        df['mean_packet_size'] = df['byte_count'] / df['packet_count']
        df['packet_size_std'] = df.groupby('flow_id')['mean_packet_size'].transform('std')
        
        # Flow entropy features
        df['flow_entropy'] = df.apply(lambda x: calculate_entropy(x['payload_bytes']), axis=1)
        
        # Protocol-specific features
        df['tcp_flags_entropy'] = df[df['protocol'] == 'TCP'].apply(
            lambda x: calculate_entropy(x['tcp_flags']), axis=1
        )
        
        return df
        
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral network features"""
        # Connection patterns
        df['unique_dst_ports'] = df.groupby('src_ip')['dst_port'].transform('nunique')
        df['connection_ratio'] = df.groupby('src_ip')['dst_ip'].transform('nunique')
        
        # Traffic patterns
        df['incoming_outgoing_ratio'] = df['incoming_bytes'] / df['outgoing_bytes']
        df['avg_payload_length'] = df['total_payload'] / df['packet_count']
        
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
        
        for feature in numeric_features:
            scaler_type = self.config['scaling'].get(feature, 'standard')
            if scaler_type in self.scalers:
                df[feature] = self.scalers[scaler_type].fit_transform(
                    df[feature].values.reshape(-1, 1)
                )
        
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        try:
            df = self.create_time_features(df)
            df = self.create_statistical_features(df)
            df = self.create_behavioral_features(df)
            df = self.scale_features(df)
            
            logger.info(f"Engineered features shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            raise