import pandas as pd
from typing import List
import numpy as np
from math import log2

def validate_dataset(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate if DataFrame contains all required columns
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    return True

def ip2long(ip: str) -> int:
    """Convert an IP address to a number"""
    try:
        ip_parts = ip.split('.')
        return sum(int(part) << (24 - 8 * i) for i, part in enumerate(ip_parts))
    except:
        return 0

def calculate_entropy(data) -> float:
    """Calculate Shannon entropy of the data"""
    try:
        if isinstance(data, (int, float)):
            return 0.0
            
        # Convert to bytes if string
        if isinstance(data, str):
            data = bytes(data, 'utf-8')
            
        # Convert to bytes if needed
        if not isinstance(data, bytes):
            data = bytes(str(data), 'utf-8')
            
        # Calculate frequency of bytes
        freq = {}
        for byte in data:
            freq[byte] = freq.get(byte, 0) + 1
            
        # Calculate entropy
        length = len(data)
        return -sum(count/length * log2(count/length) for count in freq.values())
        
    except Exception as e:
        print(f"Error calculating entropy: {str(e)}")
        return 0.0