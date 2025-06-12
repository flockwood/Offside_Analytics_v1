# tests/test_player_analyzer.py
"""Tests for player analyzer module"""

import unittest
import pandas as pd
import numpy as np
from player_analyzer import PlayerAnalyzer


class TestPlayerAnalyzer(unittest.TestCase):
    
    def setUp(self):
        # Create test dataframe with various players
        self.test_df = pd.DataFrame({
            'web_name': ['Salah', 'Kane', 'De Bruyne', 'Van Dijk', 'Alisson'],
            'team_name': ['Liverpool', 'Tottenham', 'Man City', 'Liverpool', 'Liverpool'],
            'position': ['MID', 'FWD', 'MID', 'DEF', 'GKP'],
            'minutes_played': [2000, 1800, 1500, 2100, 1900],
            'goals': [15, 20, 8, 2, 0],
            'assists': [10, 5, 15, 1, 0],
            'goals_per_90': [0.675, 1.0, 0.48, 0.086, 0],
            'assists_per_90': [0.45, 0.25, 0.9, 0.043, 0],
            'xg_per_90': [0.6, 0.9, 0.4, 0.05, 0],
            'xa_per_90': [0.4, 0.2, 0.8, 0.02, 0],
            'performance_score': [180, 200, 170, 120, 100],
            'estimated_value_millions': [35, 40, 38, 25, 20],
            'value_ratio': [5.14, 5.0, 4.47, 4.8, 5.0],
            'influence_rating': [120, 130, 140, 90, 80],
            'form_rating': [7.0, 6.5, 8.0, 5.5, 6.0]
        })
        
        self.analyzer = PlayerAnalyzer(self.test_df)
    
    def test_find_undervalued_players(self):
        """Test finding undervalued players"""
        # Test with no filters
        result = self.analyzer.find_undervalued_players(top_n=3)
        self.assertEqual(len(result), 3)
        
        # Should be sorted by value_ratio descending
        self.assertGreaterEqual(result.iloc[0]['value_ratio'], result.iloc[1]['value_ratio'])
        
        # Test with position filter
        mids = self.analyzer.find_undervalued_players(position='MID', top_n=5)
        self.assertTrue(all(mids['position'] == 'MID'))
        
        # Test with minutes filter
        high_minutes = self.analyzer.find_undervalued_players(min_minutes=1900)
        self.assertTrue(all(high_minutes['minutes_played'] >= 1900))
    
    def test_find_similar_players(self):
        """Test finding similar players"""
        # Find players similar to Salah
        similar = self.analyzer.find_similar_players('Salah', top_n=2)
        
        self.assertIsNotNone(similar)
        self.assertEqual(len(similar), 1)  # Only one other MID in test data
        self.assertEqual(similar.iloc[0]['web_name'], 'De Bruyne')
        
        # Test with non-existent player
        result = self.analyzer.find_similar_players('Messi')
        self.assertIsNone(result)
    
    def test_get_position_rankings(self):
        """Test position rankings"""
        # Get top forwards
        fwd_rankings = self.analyzer.get_position_rankings('FWD', top_n=5)
        self.assertEqual(len(fwd_rankings), 1)  # Only one FWD in test data
        self.assertEqual(fwd_rankings.iloc[0]['web_name'], 'Kane')
        
        # Test with different metric
        mid_by_assists = self.analyzer.get_position_rankings(
            'MID', metric='assists', top_n=2
        )
        self.assertEqual(mid_by_assists.iloc[0]['web_name'], 'De Bruyne')
    
    def test_recommend_transfers(self):
        """Test transfer recommendations"""
        # Get recommendations for MID and FWD
        recommendations = self.analyzer.recommend_transfers(
            ['MID', 'FWD'], budget=50.0
        )
        
        self.assertIn('MID', recommendations)
        self.assertIn('FWD', recommendations)
        
        # Check that recommendations respect budget
        for position, players in recommendations.items():
            for player in players:
                self.assertLessEqual(player['value'], 50.0)
        
        # Test with lower budget
        budget_recs = self.analyzer.recommend_transfers(
            ['DEF'], budget=30.0
        )
        for player in budget_recs['DEF']:
            self.assertLessEqual(player['value'], 30.0)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty position
        empty_result = self.analyzer.get_position_rankings('XXX')
        self.assertEqual(len(empty_result), 0)
        
        # Very high minutes filter
        no_players = self.analyzer.find_undervalued_players(min_minutes=10000)
        self.assertEqual(len(no_players), 0)
        
        # Recommendations with no valid players
        no_budget = self.analyzer.recommend_transfers(['FWD'], budget=1.0)
        self.assertEqual(len(no_budget['FWD']), 0)


if __name__ == '__main__':
    unittest.main()