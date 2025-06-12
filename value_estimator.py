# value_estimator.py
"""Module for estimating player market values"""

import pandas as pd
import numpy as np
from config import BASE_VALUES


class ValueEstimator:
    """Estimates player market values based on performance"""
    
    def __init__(self, base_values: Dict[str, float] = None):
        self.base_values = base_values or BASE_VALUES
    
    def estimate_market_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate market value for all players"""
        df = df.copy()
        df['estimated_value_millions'] = df.apply(
            lambda row: self._calculate_player_value(row), axis=1
        )
        return df
    
    def calculate_value_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate value-related metrics"""
        df = df.copy()
        
        # Value ratio (performance per million)
        df['value_ratio'] = np.where(
            df['estimated_value_millions'] > 0,
            df['performance_score'] / df['estimated_value_millions'],
            0
        )
        
        # Performance per million
        df['performance_per_million'] = df['value_ratio']
        
        # Goals per million
        df['goals_per_million'] = np.where(
            df['estimated_value_millions'] > 0,
            df['goals'] / df['estimated_value_millions'],
            0
        )
        
        return df
    
    def _calculate_player_value(self, player: pd.Series) -> float:
        """Calculate individual player market value"""
        # Handle missing position
        if pd.isna(player['position']) or player['position'] not in self.base_values:
            return 10.0
            
        base = self.base_values[player['position']]
        
        # Performance multiplier
        perf_multiplier = 1 + (player['performance_score'] / 100)
        
        # Playing time multiplier
        playing_multiplier = min(player['minutes_played'] / 2000, 1.5)
        
        # Form multiplier
        form_multiplier = 1 + (player['form_rating'] / 10)
        
        # Calculate estimated value
        value = base * perf_multiplier * playing_multiplier * form_multiplier
        
        return round(value, 1)