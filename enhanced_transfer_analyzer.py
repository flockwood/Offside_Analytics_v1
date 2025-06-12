# enhanced_transfer_analyzer.py
"""Enhanced analyzer combining FPL and Transfermarkt data"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from data_fetcher import DataFetcher
from transfermarkt_fetcher import TransfermarktFetcher
from performance_calculator import PerformanceCalculator
from value_estimator import ValueEstimator
from player_analyzer import PlayerAnalyzer


class EnhancedTransferAnalyzer:
    """Combines FPL performance data with Transfermarkt market values"""
    
    def __init__(self, apify_token: str):
        # Initialize both data sources
        self.fpl_fetcher = DataFetcher()
        self.tm_fetcher = TransfermarktFetcher(apify_token)
        
        # Initialize processors
        self.performance_calculator = PerformanceCalculator()
        self.value_estimator = ValueEstimator()
        
        # Data storage
        self.fpl_data = None
        self.tm_data = None
        self.combined_data = None
        
    def initialize(self):
        """Fetch and combine data from both sources"""
        print("Fetching FPL data...")
        self.fpl_data = self.fpl_fetcher.get_players_dataframe()
        
        # Calculate FPL metrics
        self.fpl_data = self.performance_calculator.calculate_per_90_metrics(self.fpl_data)
        self.fpl_data = self.performance_calculator.calculate_performance_scores(self.fpl_data)
        self.fpl_data = self.performance_calculator.calculate_form_metrics(self.fpl_data)
        
        print("Data initialized. Use fetch_transfermarkt_data() for specific players.")
        
    def fetch_transfermarkt_data(self, player_urls: List[str]):
        """Fetch Transfermarkt data for specific players"""
        print("Fetching Transfermarkt data...")
        self.tm_data = self.tm_fetcher.fetch_player_data(player_urls)
        
        # Combine datasets
        self._combine_data_sources()
        
    def _combine_data_sources(self):
        """Merge FPL and Transfermarkt data"""
        if self.fpl_data is None or self.tm_data is None:
            print("Both data sources must be loaded first")
            return
        
        # Create a mapping based on player names (simplified - could use fuzzy matching)
        self.combined_data = self.fpl_data.copy()
        
        # Add Transfermarkt data
        for _, tm_player in self.tm_data.iterrows():
            # Find matching player in FPL data
            matches = self.combined_data[
                self.combined_data['web_name'].str.contains(tm_player['name'], case=False, na=False)
            ]
            
            if len(matches) > 0:
                idx = matches.index[0]
                self.combined_data.loc[idx, 'tm_market_value'] = tm_player['market_value']
                self.combined_data.loc[idx, 'tm_age'] = tm_player['age']
                self.combined_data.loc[idx, 'tm_contract_until'] = tm_player['contract_until']
                self.combined_data.loc[idx, 'transfermarkt_url'] = tm_player['profile_url']
    
    def analyze_value_discrepancies(self, min_minutes: int = 900) -> pd.DataFrame:
        """Find players with big differences between estimated and actual market value"""
        if self.combined_data is None:
            print("No combined data available. Run fetch_transfermarkt_data() first.")
            return pd.DataFrame()
        
        # Filter active players with both values
        analysis_df = self.combined_data[
            (self.combined_data['minutes_played'] >= min_minutes) &
            (self.combined_data['tm_market_value'].notna())
        ].copy()
        
        # Calculate value metrics
        analysis_df['fpl_estimated_value'] = self.value_estimator._calculate_player_value(
            analysis_df, self.value_estimator.base_values
        )
        
        # Value difference
        analysis_df['value_difference'] = (
            analysis_df['tm_market_value'] - analysis_df['fpl_estimated_value']
        )
        
        # Percentage difference
        analysis_df['value_diff_pct'] = (
            (analysis_df['value_difference'] / analysis_df['tm_market_value']) * 100
        )
        
        # Performance per actual market value
        analysis_df['performance_per_tm_million'] = (
            analysis_df['performance_score'] / analysis_df['tm_market_value']
        )
        
        # Select and sort
        display_cols = [
            'web_name', 'team_name', 'position', 
            'tm_market_value', 'fpl_estimated_value', 'value_difference',
            'performance_score', 'performance_per_tm_million',
            'goals', 'assists', 'tm_age'
        ]
        
        return analysis_df[display_cols].sort_values(
            'performance_per_tm_million', ascending=False
        )
    
    def find_contract_opportunities(self) -> pd.DataFrame:
        """Find players with expiring contracts who are performing well"""
        if self.combined_data is None:
            return pd.DataFrame()
        
        # Filter for players with contract info
        contract_df = self.combined_data[
            self.combined_data['tm_contract_until'].notna()
        ].copy()
        
        # Parse contract dates (simplified)
        current_year = pd.Timestamp.now().year
        contract_df['contract_year'] = pd.to_datetime(
            contract_df['tm_contract_until'], errors='coerce'
        ).dt.year
        
        # Find expiring contracts (within 1 year)
        expiring = contract_df[
            (contract_df['contract_year'] <= current_year + 1) &
            (contract_df['performance_score'] > contract_df['performance_score'].median())
        ]
        
        display_cols = [
            'web_name', 'team_name', 'position', 'tm_contract_until',
            'tm_market_value', 'performance_score', 'goals', 'assists'
        ]
        
        return expiring[display_cols].sort_values('performance_score', ascending=False)
    
    def analyze_injuries_impact(self, injury_url: str) -> Dict:
        """Analyze how injuries affect team dynamics and find replacements"""
        # Fetch current injuries
        injuries_df = self.tm_fetcher.fetch_injury_data(injury_url)
        
        analysis = {
            'injured_players': [],
            'replacement_suggestions': {}
        }
        
        for _, injury in injuries_df.iterrows():
            player_name = injury['player_name']
            
            # Find player in combined data
            player_data = self.combined_data[
                self.combined_data['web_name'].str.contains(player_name, case=False, na=False)
            ]
            
            if not player_data.empty:
                player = player_data.iloc[0]
                
                # Find similar players who could replace them
                analyzer = PlayerAnalyzer(self.combined_data)
                similar = analyzer.find_similar_players(player['web_name'], top_n=3)
                
                analysis['injured_players'].append({
                    'name': player_name,
                    'team': injury['team'],
                    'position': player['position'],
                    'injury': injury['injury_type'],
                    'return': injury['expected_return'],
                    'performance_score': player['performance_score']
                })
                
                if similar is not None:
                    analysis['replacement_suggestions'][player_name] = similar.to_dict('records')
        
        return analysis
    
    def generate_comprehensive_report(self, player_urls: List[str]) -> str:
        """Generate a detailed report combining all data sources"""
        # Fetch Transfermarkt data
        self.fetch_transfermarkt_data(player_urls)
        
        # Analyze value discrepancies
        value_analysis = self.analyze_value_discrepancies()
        
        report = f"""
COMPREHENSIVE TRANSFER ANALYSIS REPORT
=====================================

DATA SOURCES:
- FPL: {len(self.fpl_data)} players analyzed
- Transfermarkt: {len(self.tm_data)} players with market values

TOP VALUE OPPORTUNITIES:
------------------------
Players performing above their market value:

"""
        
        # Add top undervalued players
        top_value = value_analysis.head(10)
        for _, player in top_value.iterrows():
            report += f"\n{player['web_name']} ({player['team_name']}) - {player['position']}"
            report += f"\n  Market Value: €{player['tm_market_value']}m"
            report += f"\n  Performance/Million: {player['performance_per_tm_million']:.2f}"
            report += f"\n  Stats: {player['goals']}g, {player['assists']}a"
            report += f"\n  Age: {player['tm_age']}\n"
        
        # Add contract opportunities
        contracts = self.find_contract_opportunities()
        if not contracts.empty:
            report += "\n\nCONTRACT EXPIRY OPPORTUNITIES:\n"
            report += "-------------------------------\n"
            for _, player in contracts.head(5).iterrows():
                report += f"{player['web_name']} - Contract until: {player['tm_contract_until']}\n"
        
        return report
    
    def compare_league_values(self, league_url: str) -> pd.DataFrame:
        """Compare FPL performance scores with Transfermarkt market values by team"""
        # Fetch league data from Transfermarkt
        league_data = self.tm_fetcher.fetch_league_data(league_url)
        
        # Aggregate by team
        team_comparison = self.combined_data.groupby('team_name').agg({
            'performance_score': 'sum',
            'tm_market_value': 'sum',
            'goals': 'sum',
            'assists': 'sum'
        }).reset_index()
        
        # Calculate efficiency
        team_comparison['performance_per_million'] = (
            team_comparison['performance_score'] / team_comparison['tm_market_value']
        )
        
        return team_comparison.sort_values('performance_per_million', ascending=False)


# Example usage
if __name__ == "__main__":
    import os
    
    # Set your Apify token
    analyzer = EnhancedTransferAnalyzer(os.environ.get('APIFY_API_TOKEN'))
    
    # Initialize with FPL data
    analyzer.initialize()
    
    # Example: Analyze specific players
    player_urls = [
        "https://www.transfermarkt.com/mohamed-salah/profil/spieler/148455",
        "https://www.transfermarkt.com/erling-haaland/profil/spieler/418560",
        "https://www.transfermarkt.com/bukayo-saka/profil/spieler/433177"
    ]
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(player_urls)
    print(report)
    
    # Find value opportunities
    print("\nTop Value Opportunities:")
    print("=" * 60)
    value_opportunities = analyzer.analyze_value_discrepancies()
    print(value_opportunities.head(10))
    
    # Check injuries impact
    injury_url = "https://www.transfermarkt.com/premier-league/verletztespieler/wettbewerb/GB1"
    injuries = analyzer.analyze_injuries_impact(injury_url)
    
    print("\nCurrent Injuries & Replacements:")
    print("=" * 60)
    for injury in injuries['injured_players']:
        print(f"\n{injury['name']} ({injury['team']}) - {injury['injury']}")
        print(f"Expected return: {injury['return']}")
        if injury['name'] in injuries['replacement_suggestions']:
            print("Suggested replacements:")
            for replacement in injuries['replacement_suggestions'][injury['name']][:3]:
                print(f"  - {replacement['web_name']} ({replacement['team_name']})")