# transfermarkt_fetcher.py
"""Module for fetching data from Transfermarkt via Apify"""

from apify_client import ApifyClient
import pandas as pd
from typing import Dict, List, Optional
import os
from datetime import datetime
import json

class TransfermarktFetcher:
    """Fetches real market data from Transfermarkt"""
    
    def __init__(self, api_token: str = None):
        """Initialize with Apify API token"""
        self.api_token = api_token or os.environ.get('APIFY_API_TOKEN')
        if not self.api_token:
            raise ValueError("Apify API token required. Set APIFY_API_TOKEN environment variable.")
        
        self.client = ApifyClient(self.api_token)
        self.actor_id = "curious_coder/transfermarkt"
        self._cache = {}
        
    def fetch_player_data(self, player_urls: List[str]) -> pd.DataFrame:
        """Fetch detailed player data from Transfermarkt"""
        run_input = {
            "startUrls": [{"url": url} for url in player_urls],
            "proxyConfig": {"useApifyProxy": True},
            "crawlDepth": 1,
            "pageDepth": 1,
        }
        
        # Run the actor
        print(f"Fetching data for {len(player_urls)} players from Transfermarkt...")
        run = self.client.actor(self.actor_id).call(run_input=run_input)
        
        # Collect results
        players_data = []
        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            players_data.append(self._process_player_item(item))
        
        return pd.DataFrame(players_data)
    
    def fetch_league_data(self, league_url: str) -> Dict:
        """Fetch league table and statistics"""
        run_input = {
            "startUrls": [{"url": league_url}],
            "proxyConfig": {"useApifyProxy": True},
            "crawlDepth": 2,  # Get more data from league pages
            "pageDepth": 2,
        }
        
        run = self.client.actor(self.actor_id).call(run_input=run_input)
        
        league_data = {
            'teams': [],
            'players': [],
            'standings': []
        }
        
        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            if 'table' in item.get('url', ''):
                league_data['standings'] = self._process_standings(item)
            elif 'profil/spieler' in item.get('url', ''):
                league_data['players'].append(self._process_player_item(item))
        
        return league_data
    
    def fetch_injury_data(self, injury_url: str) -> pd.DataFrame:
        """Fetch current injury list"""
        run_input = {
            "startUrls": [{"url": injury_url}],
            "proxyConfig": {"useApifyProxy": True},
            "crawlDepth": 1,
            "pageDepth": 1,
        }
        
        run = self.client.actor(self.actor_id).call(run_input=run_input)
        
        injuries = []
        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            injuries.extend(self._process_injury_data(item))
        
        return pd.DataFrame(injuries)
    
    def fetch_transfer_history(self, player_url: str) -> List[Dict]:
        """Fetch player's transfer history"""
        # Modify URL to get transfer page
        transfer_url = player_url.replace('/profil/', '/transfers/')
        
        run_input = {
            "startUrls": [{"url": transfer_url}],
            "proxyConfig": {"useApifyProxy": True},
            "crawlDepth": 1,
            "pageDepth": 1,
        }
        
        run = self.client.actor(self.actor_id).call(run_input=run_input)
        
        transfers = []
        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            transfers.extend(self._process_transfer_history(item))
        
        return transfers
    
    def _process_player_item(self, item: Dict) -> Dict:
        """Process raw player data from Transfermarkt"""
        # Extract key information
        player_data = {
            'name': item.get('name', ''),
            'full_name': item.get('fullName', ''),
            'position': item.get('position', ''),
            'age': self._parse_age(item.get('age', '')),
            'nationality': item.get('nationality', ''),
            'current_club': item.get('club', ''),
            'market_value': self._parse_market_value(item.get('marketValue', '')),
            'contract_until': item.get('contractUntil', ''),
            'height': self._parse_height(item.get('height', '')),
            'foot': item.get('foot', ''),
            'agent': item.get('agent', ''),
            'transfermarkt_id': self._extract_id_from_url(item.get('url', '')),
            'profile_url': item.get('url', ''),
            'image_url': item.get('imageUrl', ''),
            'last_updated': datetime.now().isoformat()
        }
        
        # Add performance data if available
        if 'performanceData' in item:
            perf = item['performanceData']
            player_data.update({
                'appearances': perf.get('appearances', 0),
                'goals': perf.get('goals', 0),
                'assists': perf.get('assists', 0),
                'yellow_cards': perf.get('yellowCards', 0),
                'red_cards': perf.get('redCards', 0),
                'minutes_played': perf.get('minutesPlayed', 0)
            })
        
        return player_data
    
    def _process_injury_data(self, item: Dict) -> List[Dict]:
        """Process injury list data"""
        injuries = []
        
        if 'injuries' in item:
            for injury in item['injuries']:
                injuries.append({
                    'player_name': injury.get('playerName', ''),
                    'player_id': injury.get('playerId', ''),
                    'team': injury.get('team', ''),
                    'injury_type': injury.get('injury', ''),
                    'injured_since': injury.get('since', ''),
                    'expected_return': injury.get('expectedReturn', ''),
                    'status': injury.get('status', '')
                })
        
        return injuries
    
    def _process_transfer_history(self, item: Dict) -> List[Dict]:
        """Process transfer history data"""
        transfers = []
        
        if 'transfers' in item:
            for transfer in item['transfers']:
                transfers.append({
                    'date': transfer.get('date', ''),
                    'from_club': transfer.get('from', ''),
                    'to_club': transfer.get('to', ''),
                    'transfer_fee': self._parse_market_value(transfer.get('fee', '')),
                    'market_value_at_time': self._parse_market_value(transfer.get('marketValue', '')),
                    'loan': transfer.get('loan', False)
                })
        
        return transfers
    
    def _process_standings(self, item: Dict) -> List[Dict]:
        """Process league standings data"""
        standings = []
        
        if 'standings' in item:
            for team in item['standings']:
                standings.append({
                    'position': team.get('position', 0),
                    'team': team.get('team', ''),
                    'played': team.get('played', 0),
                    'won': team.get('won', 0),
                    'drawn': team.get('drawn', 0),
                    'lost': team.get('lost', 0),
                    'goals_for': team.get('goalsFor', 0),
                    'goals_against': team.get('goalsAgainst', 0),
                    'goal_difference': team.get('goalDifference', 0),
                    'points': team.get('points', 0)
                })
        
        return standings
    
    def _parse_market_value(self, value_str: str) -> float:
        """Convert market value string to float (in millions)"""
        if not value_str:
            return 0.0
        
        # Remove currency symbols and convert
        value_str = value_str.replace('€', '').replace('£', '').replace('$', '')
        
        if 'm' in value_str.lower():
            # Million
            return float(value_str.lower().replace('m', '').strip())
        elif 'k' in value_str.lower():
            # Thousand
            return float(value_str.lower().replace('k', '').strip()) / 1000
        else:
            try:
                return float(value_str.strip())
            except:
                return 0.0
    
    def _parse_age(self, age_str: str) -> int:
        """Extract age from string"""
        try:
            # Usually in format "24 years"
            return int(age_str.split()[0])
        except:
            return 0
    
    def _parse_height(self, height_str: str) -> float:
        """Convert height string to float (in cm)"""
        try:
            # Usually in format "1,83 m" or "183 cm"
            if 'm' in height_str and ',' in height_str:
                return float(height_str.replace('m', '').replace(',', '.').strip()) * 100
            elif 'cm' in height_str:
                return float(height_str.replace('cm', '').strip())
            else:
                return 0.0
        except:
            return 0.0
    
    def _extract_id_from_url(self, url: str) -> str:
        """Extract player ID from Transfermarkt URL"""
        try:
            # URL format: .../spieler/12345
            parts = url.split('/')
            if 'spieler' in parts:
                idx = parts.index('spieler')
                return parts[idx + 1]
        except:
            pass
        return ''
    
    def build_player_urls(self, player_names: List[str], league: str = "premier-league") -> List[str]:
        """Helper to build Transfermarkt URLs from player names"""
        # This is a simplified version - in practice you'd want to search for exact URLs
        base_url = "https://www.transfermarkt.com"
        urls = []
        
        # You could implement a search function here or maintain a mapping
        print(f"Note: URL building requires exact Transfermarkt URLs or a search implementation")
        
        return urls