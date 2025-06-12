# performance_calculator.py
"""Module for calculating player performance metrics"""

import pandas as pd
import numpy as np
from typing import Dict
from config import POSITION_WEIGHTS


class PerformanceCalculator:
    """Calculates various performance metrics for players"""
    
    @staticmethod
    def calculate_per_90_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per 90 minute metrics"""
        df = df.copy()
        
        # Goals per 90
        df['goals_per_90'] = np.where(
            df['minutes_played'] > 0,
            (df['goals'] / df['minutes_played']) * 90,
            0
        )
        
        # Assists per 90
        df['assists_per_90'] = np.where(
            df['minutes_played'] > 0,
            (df['assists'] / df['minutes_played']) * 90,
            0
        )
        
        # Goal contributions per 90
        df['goal_contributions_per_90'] = df['goals_per_90'] + df['assists_per_90']
        
        # Expected goals per 90
        df['xg_per_90'] = np.where(
            df['minutes_played'] > 0,
            (df['expected_goals'].astype(float) / df['minutes_played']) * 90,
            0
        )
        
        # Expected assists per 90
        df['xa_per_90'] = np.where(
            df['minutes_played'] > 0,
            (df['expected_assists'].astype(float) / df['minutes_played']) * 90,
            0
        )
        
        return df
    
    @staticmethod
    def calculate_performance_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position-adjusted performance scores"""
        df = df.copy()
        df['performance_score'] = 0.0
        
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            mask = df['position'] == pos
            
            if pos == 'GKP':
                df.loc[mask, 'performance_score'] = (
                    df.loc[mask, 'clean_sheets'] * 10 +
                    df.loc[mask, 'saves'] * 0.5
                ).astype(float)
            elif pos == 'DEF':
                df.loc[mask, 'performance_score'] = (
                    df.loc[mask, 'clean_sheets'] * 5 +
                    df.loc[mask, 'goals'] * 15 +
                    df.loc[mask, 'assists'] * 10
                ).astype(float)
            elif pos == 'MID':
                df.loc[mask, 'performance_score'] = (
                    df.loc[mask, 'goals'] * 12 +
                    df.loc[mask, 'assists'] * 10 +
                    df.loc[mask, 'creativity'].astype(float) * 0.5
                ).astype(float)
            else:  # FWD
                df.loc[mask, 'performance_score'] = (
                    df.loc[mask, 'goals'] * 10 +
                    df.loc[mask, 'assists'] * 8 +
                    df.loc[mask, 'expected_goals'].astype(float) * 5
                ).astype(float)
                
        return df
    
    @staticmethod
    def calculate_form_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate form-based metrics"""
        df = df.copy()
        df['form_rating'] = df['form'].astype(float)
        df['influence_rating'] = df['ict_index'].astype(float) / 10
        return df