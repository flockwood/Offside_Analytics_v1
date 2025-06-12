# tests/test_data_fetcher.py
"""Tests for data fetcher module"""

import unittest
from unittest.mock import patch, Mock
import pandas as pd
from data_fetcher import DataFetcher


class TestDataFetcher(unittest.TestCase):
    
    def setUp(self):
        self.fetcher = DataFetcher()
    
    @patch('requests.get')
    def test_fetch_bootstrap_data(self, mock_get):
        """Test fetching bootstrap data"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'elements': [
                {'id': 1, 'web_name': 'Salah', 'team': 1},
                {'id': 2, 'web_name': 'Kane', 'team': 2}
            ],
            'teams': [
                {'id': 1, 'name': 'Liverpool'},
                {'id': 2, 'name': 'Tottenham'}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Test
        data = self.fetcher.fetch_bootstrap_data()
        
        self.assertIn('elements', data)
        self.assertIn('teams', data)
        self.assertEqual(len(data['elements']), 2)
    
    @patch('requests.get')
    def test_get_players_dataframe(self, mock_get):
        """Test getting players dataframe"""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'elements': [
                {
                    'id': 1, 
                    'web_name': 'Salah', 
                    'team': 1,
                    'element_type': 3,
                    'goals_scored': 10,
                    'assists': 5,
                    'minutes': 900
                }
            ],
            'teams': [
                {'id': 1, 'name': 'Liverpool'}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        # Test
        df = self.fetcher.get_players_dataframe()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['web_name'], 'Salah')
        self.assertEqual(df.iloc[0]['team_name'], 'Liverpool')
        self.assertEqual(df.iloc[0]['position'], 'MID')
    
    def test_cache_functionality(self):
        """Test that caching works properly"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {'test': 'data'}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # First call
            data1 = self.fetcher.fetch_bootstrap_data()
            # Second call (should use cache)
            data2 = self.fetcher.fetch_bootstrap_data()
            
            # Should only make one request
            self.assertEqual(mock_get.call_count, 1)
            self.assertEqual(data1, data2)


if __name__ == '__main__':
    unittest.main()