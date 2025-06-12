# data_fetcher.py
"""Module for fetching and processing FPL data"""

import requests
import pandas as pd
from typing import Dict, Optional
from config import API_CONFIG, POSITION_MAPPING


class DataFetcher:
    """Handles all data fetching and initial processing"""
    
    def __init__(self):
        self.base_url = API_CONFIG['base_url']
        self.endpoints = API_CONFIG['endpoints']
        self._cache = {}
        
    def fetch_bootstrap_data(self, use_cache: bool = True) -> Dict:
        """Fetch bootstrap data with optional caching"""
        if use_cache and 'bootstrap' in self._cache:
            return self._cache['bootstrap']
            
        url = f"{self.base_url}{self.endpoints['bootstrap']}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        self._cache['bootstrap'] = data
        return data
    
    def fetch_fixtures(self, use_cache: bool = True) -> Dict:
        """Fetch fixtures data"""
        if use_cache and 'fixtures' in self._cache:
            return self._cache['fixtures']
            
        url = f"{self.base_url}{self.endpoints['fixtures']}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        self._cache['fixtures'] = data
        return data
    
    def fetch_player_details(self, player_id: int) -> Dict:
        """Fetch detailed data for a specific player"""
        url = f"{self.base_url}{self.endpoints['player_detail'].format(player_id=player_id)}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_players_dataframe(self) -> pd.DataFrame:
        """Get processed players DataFrame"""
        bootstrap_data = self.fetch_bootstrap_data()
        players_df = pd.DataFrame(bootstrap_data['elements'])
        teams_df = pd.DataFrame(bootstrap_data['teams'])
        
        # Process data
        players_df = self._process_players_data(players_df, teams_df)
        return players_df
    
    def get_teams_dataframe(self) -> pd.DataFrame:
        """Get teams DataFrame"""
        bootstrap_data = self.fetch_bootstrap_data()
        return pd.DataFrame(bootstrap_data['teams'])
    
    def _process_players_data(self, players_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Process raw player data"""
        # Add team names
        team_dict = dict(zip(teams_df['id'], teams_df['name']))
        players_df['team_name'] = players_df['team'].map(team_dict)
        
        # Map positions
        players_df['position'] = players_df['element_type'].map(POSITION_MAPPING)
        
        # Add derived columns
        players_df['goals'] = players_df['goals_scored']
        players_df['assists'] = players_df['assists']
        players_df['minutes_played'] = players_df['minutes']
        players_df['games_played'] = players_df['minutes'] / 90
        
        return players_df