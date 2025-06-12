# tests/test_performance_calculator.py
"""Tests for performance calculator module"""

import unittest
import pandas as pd
import numpy as np
from performance_calculator import PerformanceCalculator


class TestPerformanceCalculator(unittest.TestCase):
    
    def setUp(self):
        self.calculator = PerformanceCalculator()
        
        # Create test dataframe
        self.test_df = pd.DataFrame({
            'web_name': ['Player1', 'Player2', 'Player3'],
            'position': ['FWD', 'MID', 'DEF'],
            'goals': [10, 5, 1],
            'assists': [3, 8, 2],
            'minutes_played': [900, 1800, 0],
            'expected_goals': ['8.5', '4.2', '0.5'],
            'expected_assists': ['2.8', '7.1', '1.5'],
            'clean_sheets': [0, 2, 5],
            'saves': [0, 0, 0],
            'creativity': ['50.0', '120.0', '20.0'],
            'form': ['6.5', '5.0', '3.0'],
            'ict_index': ['100.0', '150.0', '80.0']
        })
    
    def test_calculate_per_90_metrics(self):
        """Test per 90 minute calculations"""
        result = self.calculator.calculate_per_90_metrics(self.test_df)
        
        # Player1: 10 goals in 900 minutes = 1.0 per 90
        self.assertAlmostEqual(result.iloc[0]['goals_per_90'], 1.0, places=2)
        
        # Player2: 8 assists in 1800 minutes = 0.4 per 90
        self.assertAlmostEqual(result.iloc[1]['assists_per_90'], 0.4, places=2)
        
        # Player3: 0 minutes should result in 0 per 90 stats
        self.assertEqual(result.iloc[2]['goals_per_90'], 0)
    
    def test_calculate_performance_scores(self):
        """Test position-adjusted performance scores"""
        result = self.calculator.calculate_performance_scores(self.test_df)
        
        # Check that performance scores are calculated
        self.assertIn('performance_score', result.columns)
        
        # FWD should have high score due to goals
        self.assertGreater(result.iloc[0]['performance_score'], 0)
        
        # All scores should be floats
        self.assertTrue(all(isinstance(score, float) for score in result['performance_score']))
    
    def test_calculate_form_metrics(self):
        """Test form metric calculations"""
        result = self.calculator.calculate_form_metrics(self.test_df)
        
        # Check form rating
        self.assertEqual(result.iloc[0]['form_rating'], 6.5)
        self.assertEqual(result.iloc[1]['form_rating'], 5.0)
        
        # Check influence rating (ict_index / 10)
        self.assertEqual(result.iloc[0]['influence_rating'], 10.0)
        self.assertEqual(result.iloc[1]['influence_rating'], 15.0)
    
    def test_edge_cases(self):
        """Test edge cases like zero minutes, missing data"""
        edge_df = pd.DataFrame({
            'web_name': ['NoMinutes', 'MissingData'],
            'position': ['FWD', 'MID'],
            'goals': [5, np.nan],
            'assists': [2, 3],
            'minutes_played': [0, 1000],
            'expected_goals': ['0', ''],
            'expected_assists': ['0', '2.0'],
            'clean_sheets': [0, 0],
            'saves': [0, 0],
            'creativity': ['0', '50.0'],
            'form': ['0', '4.0'],
            'ict_index': ['0', '100.0']
        })
        
        # Should handle without errors
        result = self.calculator.calculate_per_90_metrics(edge_df)
        
        # Player with 0 minutes should have 0 for per 90 stats
        self.assertEqual(result.iloc[0]['goals_per_90'], 0)
        self.assertEqual(result.iloc[0]['assists_per_90'], 0)


if __name__ == '__main__':
    unittest.main()