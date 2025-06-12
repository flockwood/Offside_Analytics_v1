# report_generator.py
"""Module for generating analysis reports"""

import pandas as pd
from typing import Dict, List
from datetime import datetime


class ReportGenerator:
    """Generates text and data reports"""
    
    def __init__(self, players_df: pd.DataFrame):
        self.players_df = players_df
    
    def generate_market_summary(self) -> str:
        """Generate market analysis summary"""
        active_players = self.players_df[self.players_df['minutes_played'] >= 900]
        
        summary = f"""
MARKET ANALYSIS SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

Total Players Analyzed: {len(self.players_df)}
Active Players (900+ mins): {len(active_players)}

MARKET OVERVIEW:
- Average Player Value: £{active_players['estimated_value_millions'].mean():.1f}m
- Total Market Value: £{active_players['estimated_value_millions'].sum():.1f}m
- Most Valuable Player: {active_players.nlargest(1, 'estimated_value_millions')['web_name'].values[0]} (£{active_players['estimated_value_millions'].max():.1f}m)

TOP PERFORMERS:
- Top Scorer: {active_players.nlargest(1, 'goals')['web_name'].values[0]} ({active_players['goals'].max()} goals)
- Top Assister: {active_players.nlargest(1, 'assists')['web_name'].values[0]} ({active_players['assists'].max()} assists)
- Best Performance Score: {active_players.nlargest(1, 'performance_score')['web_name'].values[0]} ({active_players['performance_score'].max():.1f})

POSITION BREAKDOWN:
"""
        
        for position in ['GKP', 'DEF', 'MID', 'FWD']:
            pos_players = active_players[active_players['position'] == position]
            summary += f"- {position}: {len(pos_players)} players, avg value £{pos_players['estimated_value_millions'].mean():.1f}m\n"
        
        return summary
    
    def generate_transfer_report(self, recommendations: Dict[str, List[Dict]], 
                               budget: float) -> str:
        """Generate transfer recommendations report"""
        report = f"""
TRANSFER RECOMMENDATIONS REPORT
Budget: £{budget}m per player
{'='*60}

"""
        for position, players in recommendations.items():
            report += f"\n{position} OPTIONS:\n"
            report += "-" * 40 + "\n"
            
            for i, player in enumerate(players, 1):
                report += f"\n{i}. {player['name']} ({player['team']}) - £{player['value']}m\n"
                report += f"   Goals: {player['goals']} | Assists: {player['assists']}\n"
                report += f"   Per 90: {player['goals_per_90']}g, {player['assists_per_90']}a\n"
                report += f"   Performance Score: {player['performance_score']}\n"
        
        return report
    
    def generate_player_report(self, player_name: str) -> str:
        """Generate detailed player report"""
        player = self.players_df[self.players_df['web_name'] == player_name]
        
        if player.empty:
            return f"Player {player_name} not found"
        
        player = player.iloc[0]
        
        report = f"""
PLAYER REPORT: {player['web_name']}
{'='*60}

BASIC INFO:
- Team: {player['team_name']}
- Position: {player['position']}
- Estimated Value: £{player['estimated_value_millions']}m
- Minutes Played: {player['minutes_played']}

PERFORMANCE STATS:
- Goals: {player['goals']}
- Assists: {player['assists']}
- Goals per 90: {player['goals_per_90']:.2f}
- Assists per 90: {player['assists_per_90']:.2f}
- Performance Score: {player['performance_score']:.1f}
- Value Ratio: {player['value_ratio']:.2f}

ADVANCED METRICS:
- Expected Goals: {player['expected_goals']}
- Expected Assists: {player['expected_assists']}
- ICT Index: {player['ict_index']}
- Form Rating: {player['form_rating']:.1f}
- Influence Rating: {player['influence_rating']:.1f}
"""
        
        return report
    
    def export_to_csv(self, filename: str, data_type: str = 'all'):
        """Export data to CSV file"""
        if data_type == 'all':
            self.players_df.to_csv(filename, index=False)
        elif data_type == 'undervalued':
            undervalued = self.players_df[
                self.players_df['minutes_played'] >= 900
            ].nlargest(50, 'value_ratio')
            undervalued.to_csv(filename, index=False)
        elif data_type == 'top_performers':
            top_performers = self.players_df[
                self.players_df['minutes_played'] >= 900
            ].nlargest(50, 'performance_score')
            top_performers.to_csv(filename, index=False)
    
    def generate_team_report(self, team_name: str) -> str:
        """Generate team analysis report"""
        team_players = self.players_df[self.players_df['team_name'] == team_name]
        
        if team_players.empty:
            return f"Team {team_name} not found"
        
        report = f"""
TEAM REPORT: {team_name}
{'='*60}

SQUAD OVERVIEW:
- Total Players: {len(team_players)}
- Squad Value: £{team_players['estimated_value_millions'].sum():.1f}m
- Average Player Value: £{team_players['estimated_value_millions'].mean():.1f}m

TEAM PERFORMANCE:
- Total Goals: {team_players['goals'].sum()}
- Total Assists: {team_players['assists'].sum()}
- Average Performance Score: {team_players['performance_score'].mean():.1f}

TOP PLAYERS:
"""
        
        top_3 = team_players.nlargest(3, 'performance_score')
        for i, (_, player) in enumerate(top_3.iterrows(), 1):
            report += f"{i}. {player['web_name']} ({player['position']}) - Score: {player['performance_score']:.1f}\n"
        
        return report