# player_analyzer.py
"""Module for player analysis and recommendations"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class PlayerAnalyzer:
    """Analyzes players and provides recommendations"""
    
    def __init__(self, players_df: pd.DataFrame):
        self.players_df = players_df
    
    def find_undervalued_players(self, position: str = None, 
                                min_minutes: int = 900, 
                                max_price: float = None,
                                top_n: int = 10) -> pd.DataFrame:
        """Find undervalued players based on performance vs value"""
        df = self.players_df.copy()
        
        # Apply filters
        df = df[df['minutes_played'] >= min_minutes]
        
        if position:
            df = df[df['position'] == position]
            
        if max_price:
            df = df[df['estimated_value_millions'] <= max_price]
        
        # Sort by value ratio
        df = df.sort_values('value_ratio', ascending=False)
        
        # Select display columns
        display_cols = [
            'web_name', 'team_name', 'position', 'estimated_value_millions',
            'goals', 'assists', 'goals_per_90', 'assists_per_90', 
            'performance_score', 'value_ratio'
        ]
        
        return df[display_cols].head(top_n)
    
    def find_similar_players(self, player_name: str, 
                           similarity_metrics: List[str] = None,
                           top_n: int = 5) -> Optional[pd.DataFrame]:
        """Find players with similar profiles"""
        if similarity_metrics is None:
            similarity_metrics = [
                'goals_per_90', 'assists_per_90', 
                'xg_per_90', 'xa_per_90', 'influence_rating'
            ]
        
        # Find target player
        target = self.players_df[self.players_df['web_name'] == player_name]
        
        if target.empty:
            print(f"Player {player_name} not found")
            return None
            
        target = target.iloc[0]
        df = self.players_df[self.players_df['position'] == target['position']].copy()
        
        # Calculate similarity
        for metric in similarity_metrics:
            df[f'{metric}_diff'] = abs(df[metric] - target[metric])
            
        # Overall similarity score
        df['similarity_score'] = sum(df[f'{metric}_diff'] for metric in similarity_metrics)
        
        # Remove target player
        df = df[df['web_name'] != player_name]
        
        # Sort by similarity
        df = df.sort_values('similarity_score')
        
        # Select display columns
        display_cols = [
            'web_name', 'team_name', 'estimated_value_millions',
            'goals', 'assists', 'goals_per_90', 'assists_per_90', 
            'performance_score'
        ]
        
        return df[display_cols].head(top_n)
    
    def get_position_rankings(self, position: str, 
                            metric: str = 'performance_score',
                            min_minutes: int = 900,
                            top_n: int = 10) -> pd.DataFrame:
        """Get top players in a position by specified metric"""
        df = self.players_df[
            (self.players_df['position'] == position) &
            (self.players_df['minutes_played'] >= min_minutes)
        ].copy()
        
        df = df.sort_values(metric, ascending=False)
        
        display_cols = [
            'web_name', 'team_name', 'estimated_value_millions',
            'goals', 'assists', metric
        ]
        
        return df[display_cols].head(top_n)
    
    def recommend_transfers(self, target_positions: List[str], 
                          budget: float = 100.0,
                          min_minutes: int = 900) -> Dict[str, List[Dict]]:
        """Recommend transfer targets"""
        recommendations = {}
        
        for position in target_positions:
            # Get candidates
            candidates = self.players_df[
                (self.players_df['position'] == position) &
                (self.players_df['estimated_value_millions'] <= budget) &
                (self.players_df['minutes_played'] >= min_minutes)
            ].copy()
            
            # Sort by performance score
            candidates = candidates.sort_values('performance_score', ascending=False)
            
            # Get top 3 options
            top_candidates = candidates.head(3)
            
            recommendations[position] = []
            
            for _, player in top_candidates.iterrows():
                rec = {
                    'name': player['web_name'],
                    'team': player['team_name'],
                    'value': player['estimated_value_millions'],
                    'goals': player['goals'],
                    'assists': player['assists'],
                    'goals_per_90': round(player['goals_per_90'], 2),
                    'assists_per_90': round(player['assists_per_90'], 2),
                    'performance_score': round(player['performance_score'], 1)
                }
                recommendations[position].append(rec)
                
        return recommendations