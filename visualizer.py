# visualizer.py
"""Module for creating visualizations"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
from config import VISUALIZATION_STYLE


class Visualizer:
    """Handles all visualization tasks"""
    
    def __init__(self, style_config: dict = None):
        self.style_config = style_config or VISUALIZATION_STYLE
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style"""
        plt.style.use(self.style_config['style'])
        sns.set_palette(self.style_config['palette'])
    
    def plot_top_performers(self, df: pd.DataFrame, top_n: int = 10):
        """Create top performers visualization"""
        fig, axes = plt.subplots(2, 2, figsize=self.style_config['figure_size'])
        fig.suptitle('Top Performers Analysis', fontsize=16)
        
        # Filter active players
        active_players = df[df['minutes_played'] >= 900]
        
        # Top scorers
        self._plot_top_scorers(active_players, axes[0, 0], top_n)
        
        # Top assisters
        self._plot_top_assisters(active_players, axes[0, 1], top_n)
        
        # Goals vs Assists scatter
        self._plot_goals_vs_assists(active_players, axes[1, 0])
        
        # Performance by position
        self._plot_performance_by_position(active_players, axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def plot_value_analysis(self, df: pd.DataFrame, analyzer=None, top_n: int = 15):
        """Create value analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=self.style_config['figure_size'])
        fig.suptitle('Player Value Analysis', fontsize=16)
        
        active_players = df[df['minutes_played'] >= 900].copy()
        
        # Performance vs Value
        self._plot_performance_vs_value(active_players, axes[0, 0])
        
        # Undervalued players
        if analyzer:
            self._plot_undervalued_players(analyzer, axes[0, 1], top_n)
        
        # Goals per 90 vs Value
        self._plot_goals_vs_value(active_players, axes[1, 0])
        
        # Value distribution
        self._plot_value_distribution(active_players, axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def plot_player_radar(self, df: pd.DataFrame, player_names: List[str]):
        """Create radar chart comparing players"""
        players_data = df[df['web_name'].isin(player_names)]
        
        if len(players_data) == 0:
            print("No players found")
            return None
            
        metrics = ['goals_per_90', 'assists_per_90', 'xg_per_90', 'xa_per_90', 'influence_rating']
        
        # Normalize metrics
        normalized_data = {}
        for metric in metrics:
            max_val = df[metric].max()
            if max_val > 0:
                normalized_data[metric] = (players_data[metric] / max_val * 100).values
            else:
                normalized_data[metric] = players_data[metric].values
                
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Setup angles
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        # Plot each player
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, (_, player) in enumerate(players_data.iterrows()):
            values = [normalized_data[metric][idx] for metric in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=player['web_name'], color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 100)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Player Comparison Radar Chart', size=16, y=1.08)
        
        return fig
    
    def plot_team_analysis(self, df: pd.DataFrame):
        """Create team analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=self.style_config['figure_size'])
        fig.suptitle('Team Analysis', fontsize=16)
        
        # Aggregate by team
        team_stats = df.groupby('team_name').agg({
            'goals': 'sum',
            'assists': 'sum',
            'performance_score': 'mean',
            'estimated_value_millions': 'sum'
        }).reset_index()
        
        # Goals by team
        self._plot_team_goals(team_stats, axes[0, 0])
        
        # Squad value by team
        self._plot_squad_value(team_stats, axes[0, 1])
        
        # Goals vs Squad Value
        self._plot_goals_vs_squad_value(team_stats, axes[1, 0])
        
        # Average performance by team
        self._plot_team_performance(team_stats, axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def plot_transfer_matrix(self, df: pd.DataFrame, positions: List[str], max_value: float = 50):
        """Create transfer target matrix"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get candidates
        candidates = df[
            (df['position'].isin(positions)) &
            (df['minutes_played'] >= 900) &
            (df['estimated_value_millions'] <= max_value)
        ].copy()
        
        # Create pivot table
        pivot_data = candidates.pivot_table(
            values='performance_score',
            index='team_name',
            columns='position',
            aggfunc='max',
            fill_value=0
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, 
                   annot=True, 
                   fmt='.1f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Performance Score'},
                   ax=ax)
        
        ax.set_title(f'Best Players by Team and Position (Max Value: £{max_value}M)')
        ax.set_xlabel('Position')
        ax.set_ylabel('Team')
        
        return fig
    
    # Helper methods for specific plots
    def _plot_top_scorers(self, df: pd.DataFrame, ax, top_n: int):
        top_scorers = df.nlargest(top_n, 'goals')
        ax.barh(top_scorers['web_name'], top_scorers['goals'])
        ax.set_xlabel('Goals')
        ax.set_title(f'Top {top_n} Goal Scorers')
        ax.invert_yaxis()
    
    def _plot_top_assisters(self, df: pd.DataFrame, ax, top_n: int):
        top_assisters = df.nlargest(top_n, 'assists')
        ax.barh(top_assisters['web_name'], top_assisters['assists'], color='orange')
        ax.set_xlabel('Assists')
        ax.set_title(f'Top {top_n} Assist Providers')
        ax.invert_yaxis()
    
    def _plot_goals_vs_assists(self, df: pd.DataFrame, ax):
        for pos, color in zip(['FWD', 'MID', 'DEF', 'GKP'], ['red', 'blue', 'green', 'purple']):
            pos_data = df[df['position'] == pos]
            ax.scatter(pos_data['goals'], pos_data['assists'], 
                      alpha=0.6, label=pos, color=color, s=60)
        ax.set_xlabel('Goals')
        ax.set_ylabel('Assists')
        ax.set_title('Goals vs Assists by Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_by_position(self, df: pd.DataFrame, ax):
        position_performance = df.groupby('position')['performance_score'].mean().sort_values()
        ax.bar(position_performance.index, position_performance.values)
        ax.set_xlabel('Position')
        ax.set_ylabel('Average Performance Score')
        ax.set_title('Average Performance Score by Position')
    
    def _plot_performance_vs_value(self, df: pd.DataFrame, ax):
        for pos, color in zip(['FWD', 'MID', 'DEF', 'GKP'], ['red', 'blue', 'green', 'purple']):
            pos_data = df[df['position'] == pos]
            ax.scatter(pos_data['estimated_value_millions'], 
                      pos_data['performance_score'],
                      alpha=0.6, label=pos, color=color, s=60)
        ax.set_xlabel('Estimated Value (£M)')
        ax.set_ylabel('Performance Score')
        ax.set_title('Performance vs Market Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_undervalued_players(self, analyzer, ax, top_n: int):
        undervalued = analyzer.find_undervalued_players(top_n=top_n)
        ax.barh(undervalued['web_name'].head(10), 
               undervalued['value_ratio'].head(10), 
               color='green')
        ax.set_xlabel('Value Ratio (Performance/Price)')
        ax.set_title('Most Undervalued Players')
        ax.invert_yaxis()
    
    def _plot_goals_vs_value(self, df: pd.DataFrame, ax):
        forwards_mids = df[df['position'].isin(['FWD', 'MID'])]
        scatter = ax.scatter(forwards_mids['estimated_value_millions'], 
                           forwards_mids['goals_per_90'],
                           c=forwards_mids['assists_per_90'], 
                           cmap='viridis', s=80, alpha=0.7)
        ax.set_xlabel('Estimated Value (£M)')
        ax.set_ylabel('Goals per 90')
        ax.set_title('Attacking Output vs Value (color = assists/90)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Assists per 90')
    
    def _plot_value_distribution(self, df: pd.DataFrame, ax):
        df.boxplot(column='estimated_value_millions', by='position', ax=ax)
        ax.set_xlabel('Position')
        ax.set_ylabel('Estimated Value (£M)')
        ax.set_title('Value Distribution by Position')
    
    def _plot_team_goals(self, team_stats: pd.DataFrame, ax):
        team_stats_sorted = team_stats.sort_values('goals', ascending=True)
        ax.barh(team_stats_sorted['team_name'], team_stats_sorted['goals'])
        ax.set_xlabel('Total Goals')
        ax.set_title('Goals by Team')
    
    def _plot_squad_value(self, team_stats: pd.DataFrame, ax):
        team_stats_sorted = team_stats.sort_values('estimated_value_millions', ascending=True)
        ax.barh(team_stats_sorted['team_name'], team_stats_sorted['estimated_value_millions'])
        ax.set_xlabel('Total Squad Value (£M)')
        ax.set_title('Squad Value by Team')
    
    def _plot_goals_vs_squad_value(self, team_stats: pd.DataFrame, ax):
        ax.scatter(team_stats['estimated_value_millions'], team_stats['goals'], s=100)
        for idx, row in team_stats.iterrows():
            ax.annotate(row['team_name'][:3], 
                        (row['estimated_value_millions'], row['goals']),
                        fontsize=8, alpha=0.7)
        ax.set_xlabel('Squad Value (£M)')
        ax.set_ylabel('Total Goals')
        ax.set_title('Goals vs Squad Value')
        ax.grid(True, alpha=0.3)
    
    def _plot_team_performance(self, team_stats: pd.DataFrame, ax):
        team_perf = team_stats.sort_values('performance_score', ascending=True)
        ax.barh(team_perf['team_name'], team_perf['performance_score'])
        ax.set_xlabel('Average Performance Score')
        ax.set_title('Average Player Performance by Team')