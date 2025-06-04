import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class TransferValueAnalyzer:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.data = {}
        self.players_df = None
        self.teams_df = None
        
    def fetch_data(self):
        """Fetch player data from FPL API (using it as a data source for real stats)"""
        print("Fetching player data...")
        
        # Get bootstrap data (players, teams, etc.)
        bootstrap_url = f"{self.base_url}/bootstrap-static/"
        response = requests.get(bootstrap_url)
        self.data['bootstrap'] = response.json()
        
        # Process data into dataframes
        self._process_data()
        print("Data fetched successfully!")
        
    def _process_data(self):
        """Convert API data to pandas DataFrames"""
        # Players DataFrame
        players = self.data['bootstrap']['elements']
        self.players_df = pd.DataFrame(players)
        
        # Teams DataFrame
        teams = self.data['bootstrap']['teams']
        self.teams_df = pd.DataFrame(teams)
        
        # Add team names to players
        team_dict = dict(zip(self.teams_df['id'], self.teams_df['name']))
        self.players_df['team_name'] = self.players_df['team'].map(team_dict)
        
        # Position mapping
        pos_dict = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        self.players_df['position'] = self.players_df['element_type'].map(pos_dict)
        
        # Convert to more realistic metrics
        self.players_df['goals'] = self.players_df['goals_scored']
        self.players_df['assists'] = self.players_df['assists']
        self.players_df['minutes_played'] = self.players_df['minutes']
        self.players_df['games_played'] = self.players_df['minutes'] / 90  # Approximate
        
    def calculate_performance_metrics(self):
        """Calculate real-world performance metrics for all players"""
        print("\nCalculating performance metrics...")
        
        # Goals per 90 minutes
        self.players_df['goals_per_90'] = np.where(
            self.players_df['minutes_played'] > 0,
            (self.players_df['goals'] / self.players_df['minutes_played']) * 90,
            0
        )
        
        # Assists per 90 minutes
        self.players_df['assists_per_90'] = np.where(
            self.players_df['minutes_played'] > 0,
            (self.players_df['assists'] / self.players_df['minutes_played']) * 90,
            0
        )
        
        # Goal contributions per 90
        self.players_df['goal_contributions_per_90'] = (
            self.players_df['goals_per_90'] + self.players_df['assists_per_90']
        )
        
        # Expected goals per 90
        self.players_df['xg_per_90'] = np.where(
            self.players_df['minutes_played'] > 0,
            (self.players_df['expected_goals'].astype(float) / self.players_df['minutes_played']) * 90,
            0
        )
        
        # Expected assists per 90
        self.players_df['xa_per_90'] = np.where(
            self.players_df['minutes_played'] > 0,
            (self.players_df['expected_assists'].astype(float) / self.players_df['minutes_played']) * 90,
            0
        )
        
        # Performance score (position-adjusted)
        self._calculate_position_adjusted_scores()
        
        # Age-adjusted potential (younger players get bonus)
        # Note: FPL doesn't provide age, so we'll simulate this
        self.players_df['age_factor'] = 1.0  # Placeholder - in real analysis you'd have actual ages
        
        # Form rating (based on recent performances)
        self.players_df['form_rating'] = self.players_df['form'].astype(float)
        
        # ICT Index as creativity/influence metric
        self.players_df['influence_rating'] = self.players_df['ict_index'].astype(float) / 10
        
    def _calculate_position_adjusted_scores(self):
        """Calculate performance scores adjusted by position"""
        # Different weights for different positions
        position_weights = {
            'GKP': {'clean_sheets': 1.0, 'saves': 0.5, 'goals': 0.1, 'assists': 0.1},
            'DEF': {'clean_sheets': 0.7, 'goals': 0.8, 'assists': 0.6, 'tackles': 0.5},
            'MID': {'goals': 0.9, 'assists': 1.0, 'key_passes': 0.7, 'chances': 0.6},
            'FWD': {'goals': 1.0, 'assists': 0.7, 'shots': 0.5, 'xg': 0.8}
        }
        
        # Initialize performance score as float
        self.players_df['performance_score'] = 0.0
        
        for pos in ['GKP', 'DEF', 'MID', 'FWD']:
            mask = self.players_df['position'] == pos
            
            if pos == 'GKP':
                self.players_df.loc[mask, 'performance_score'] = (
                    self.players_df.loc[mask, 'clean_sheets'] * 10 +
                    self.players_df.loc[mask, 'saves'] * 0.5
                ).astype(float)
            elif pos == 'DEF':
                self.players_df.loc[mask, 'performance_score'] = (
                    self.players_df.loc[mask, 'clean_sheets'] * 5 +
                    self.players_df.loc[mask, 'goals'] * 15 +
                    self.players_df.loc[mask, 'assists'] * 10
                ).astype(float)
            elif pos == 'MID':
                self.players_df.loc[mask, 'performance_score'] = (
                    self.players_df.loc[mask, 'goals'] * 12 +
                    self.players_df.loc[mask, 'assists'] * 10 +
                    self.players_df.loc[mask, 'creativity'].astype(float) * 0.5
                ).astype(float)
            else:  # FWD
                self.players_df.loc[mask, 'performance_score'] = (
                    self.players_df.loc[mask, 'goals'] * 10 +
                    self.players_df.loc[mask, 'assists'] * 8 +
                    self.players_df.loc[mask, 'expected_goals'].astype(float) * 5
                ).astype(float)
                
    def estimate_market_value(self):
        """Estimate real-world market value based on performance metrics"""
        # This is a simplified model - real valuations would be much more complex
        base_values = {
            'GKP': 10,  # Base value in millions
            'DEF': 15,
            'MID': 20,
            'FWD': 25
        }
        
        self.players_df['estimated_value_millions'] = self.players_df.apply(
            lambda row: self._calculate_player_value(row, base_values), axis=1
        )
        
    def _calculate_player_value(self, player, base_values):
        """Calculate individual player market value"""
        # Handle missing position
        if pd.isna(player['position']) or player['position'] not in base_values:
            return 10.0  # Default value for players with missing position
            
        base = base_values[player['position']]
        
        # Performance multiplier
        perf_multiplier = 1 + (player['performance_score'] / 100)
        
        # Playing time multiplier
        playing_multiplier = min(player['minutes_played'] / 2000, 1.5)
        
        # Form multiplier
        form_multiplier = 1 + (player['form_rating'] / 10)
        
        # Calculate estimated value
        value = base * perf_multiplier * playing_multiplier * form_multiplier
        
    def visualize_top_performers(self, top_n: int = 10):
        """Create visualizations for top performers"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Top Performers Analysis', fontsize=16)
        
        # Filter players with significant playing time
        active_players = self.players_df[self.players_df['minutes_played'] >= 900]
        
        # 1. Top scorers
        ax1 = axes[0, 0]
        top_scorers = active_players.nlargest(top_n, 'goals')
        ax1.barh(top_scorers['web_name'], top_scorers['goals'])
        ax1.set_xlabel('Goals')
        ax1.set_title(f'Top {top_n} Goal Scorers')
        ax1.invert_yaxis()
        
        # 2. Top assisters
        ax2 = axes[0, 1]
        top_assisters = active_players.nlargest(top_n, 'assists')
        ax2.barh(top_assisters['web_name'], top_assisters['assists'], color='orange')
        ax2.set_xlabel('Assists')
        ax2.set_title(f'Top {top_n} Assist Providers')
        ax2.invert_yaxis()
        
        # 3. Goals vs Assists scatter
        ax3 = axes[1, 0]
        for pos, color in zip(['FWD', 'MID', 'DEF', 'GKP'], ['red', 'blue', 'green', 'purple']):
            pos_data = active_players[active_players['position'] == pos]
            ax3.scatter(pos_data['goals'], pos_data['assists'], 
                       alpha=0.6, label=pos, color=color, s=60)
        ax3.set_xlabel('Goals')
        ax3.set_ylabel('Assists')
        ax3.set_title('Goals vs Assists by Position')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance score by position
        ax4 = axes[1, 1]
        position_performance = active_players.groupby('position')['performance_score'].mean().sort_values()
        ax4.bar(position_performance.index, position_performance.values)
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Average Performance Score')
        ax4.set_title('Average Performance Score by Position')
        
        plt.tight_layout()
        plt.show()
        
    def visualize_value_analysis(self, top_n: int = 15):
        """Visualize value analysis and undervalued players"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Value Analysis', fontsize=16)
        
        # Filter players
        active_players = self.players_df[self.players_df['minutes_played'] >= 900].copy()
        
        # 1. Performance vs Value scatter
        ax1 = axes[0, 0]
        for pos, color in zip(['FWD', 'MID', 'DEF', 'GKP'], ['red', 'blue', 'green', 'purple']):
            pos_data = active_players[active_players['position'] == pos]
            ax1.scatter(pos_data['estimated_value_millions'], 
                       pos_data['performance_score'],
                       alpha=0.6, label=pos, color=color, s=60)
        ax1.set_xlabel('Estimated Value (£M)')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Performance vs Market Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Most undervalued players
        ax2 = axes[0, 1]
        undervalued = self.find_undervalued_players(top_n=top_n)
        ax2.barh(undervalued['web_name'].head(10), 
                undervalued['value_ratio'].head(10), 
                color='green')
        ax2.set_xlabel('Value Ratio (Performance/Price)')
        ax2.set_title('Most Undervalued Players')
        ax2.invert_yaxis()
        
        # 3. Goals per 90 vs Value
        ax3 = axes[1, 0]
        forwards_mids = active_players[active_players['position'].isin(['FWD', 'MID'])]
        scatter = ax3.scatter(forwards_mids['estimated_value_millions'], 
                            forwards_mids['goals_per_90'],
                            c=forwards_mids['assists_per_90'], 
                            cmap='viridis', s=80, alpha=0.7)
        ax3.set_xlabel('Estimated Value (£M)')
        ax3.set_ylabel('Goals per 90')
        ax3.set_title('Attacking Output vs Value (color = assists/90)')
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Assists per 90')
        
        # 4. Value distribution by position
        ax4 = axes[1, 1]
        active_players.boxplot(column='estimated_value_millions', by='position', ax=ax4)
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Estimated Value (£M)')
        ax4.set_title('Value Distribution by Position')
        
        plt.tight_layout()
        plt.show()
        
    def visualize_player_comparison(self, player_names: List[str]):
        """Create radar chart comparing multiple players"""
        # Filter players
        players_data = self.players_df[self.players_df['web_name'].isin(player_names)]
        
        if len(players_data) == 0:
            print("No players found with those names")
            return
            
        # Select metrics for radar chart
        metrics = ['goals_per_90', 'assists_per_90', 'xg_per_90', 'xa_per_90', 'influence_rating']
        
        # Normalize metrics to 0-100 scale
        normalized_data = {}
        for metric in metrics:
            max_val = self.players_df[metric].max()
            if max_val > 0:
                normalized_data[metric] = (players_data[metric] / max_val * 100).values
            else:
                normalized_data[metric] = players_data[metric].values
                
        # Create radar chart
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
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
            
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 100)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Player Comparison Radar Chart', size=16, y=1.08)
        plt.tight_layout()
        plt.show()
        
    def visualize_team_analysis(self):
        """Visualize team-level statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Team Analysis', fontsize=16)
        
        # Aggregate by team
        team_stats = self.players_df.groupby('team_name').agg({
            'goals': 'sum',
            'assists': 'sum',
            'performance_score': 'mean',
            'estimated_value_millions': 'sum'
        }).reset_index()
        
        # 1. Goals by team
        ax1 = axes[0, 0]
        team_stats_sorted = team_stats.sort_values('goals', ascending=True)
        ax1.barh(team_stats_sorted['team_name'], team_stats_sorted['goals'])
        ax1.set_xlabel('Total Goals')
        ax1.set_title('Goals by Team')
        
        # 2. Squad value by team
        ax2 = axes[0, 1]
        team_stats_sorted = team_stats.sort_values('estimated_value_millions', ascending=True)
        ax2.barh(team_stats_sorted['team_name'], team_stats_sorted['estimated_value_millions'])
        ax2.set_xlabel('Total Squad Value (£M)')
        ax2.set_title('Squad Value by Team')
        
        # 3. Goals vs Squad Value
        ax3 = axes[1, 0]
        ax3.scatter(team_stats['estimated_value_millions'], team_stats['goals'], s=100)
        for idx, row in team_stats.iterrows():
            ax3.annotate(row['team_name'][:3], 
                        (row['estimated_value_millions'], row['goals']),
                        fontsize=8, alpha=0.7)
        ax3.set_xlabel('Squad Value (£M)')
        ax3.set_ylabel('Total Goals')
        ax3.set_title('Goals vs Squad Value')
        ax3.grid(True, alpha=0.3)
        
        # 4. Average performance by team
        ax4 = axes[1, 1]
        team_perf = team_stats.sort_values('performance_score', ascending=True)
        ax4.barh(team_perf['team_name'], team_perf['performance_score'])
        ax4.set_xlabel('Average Performance Score')
        ax4.set_title('Average Player Performance by Team')
        
        plt.tight_layout()
        plt.show()
        
    def create_transfer_target_matrix(self, positions: List[str], max_value: float = 50):
        """Create a matrix visualization of potential transfer targets"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get candidates
        candidates = self.players_df[
            (self.players_df['position'].isin(positions)) &
            (self.players_df['minutes_played'] >= 900) &
            (self.players_df['estimated_value_millions'] <= max_value)
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
        
        plt.tight_layout()
        plt.show()
    
    def find_undervalued_players(self, position: str = None, min_minutes: int = 900, top_n: int = 10):
        """Find potentially undervalued players based on performance vs estimated value"""
        df = self.players_df.copy()
        
        # Filter by playing time
        df = df[df['minutes_played'] >= min_minutes]
        
        # Filter by position
        if position:
            df = df[df['position'] == position]
            
        # Calculate value ratio (performance per million estimated value)
        # Avoid division by zero
        df['value_ratio'] = np.where(
            df['estimated_value_millions'] > 0,
            df['performance_score'] / df['estimated_value_millions'],
            0
        )
        
        # Sort by value ratio
        df = df.sort_values('value_ratio', ascending=False)
        
        # Select columns to display
        display_cols = ['web_name', 'team_name', 'position', 'estimated_value_millions',
                       'goals', 'assists', 'goals_per_90', 'assists_per_90', 
                       'performance_score', 'value_ratio']
        
        return df[display_cols].head(top_n)
    
    def find_similar_players(self, player_name: str, top_n: int = 5):
        """Find players with similar profiles for potential replacements"""
        # Find the target player
        target = self.players_df[self.players_df['web_name'] == player_name]
        
        if target.empty:
            print(f"Player {player_name} not found")
            return None
            
        target = target.iloc[0]
        df = self.players_df[self.players_df['position'] == target['position']].copy()
        
        # Calculate similarity based on key metrics
        metrics = ['goals_per_90', 'assists_per_90', 'xg_per_90', 'xa_per_90', 'influence_rating']
        
        for metric in metrics:
            df[f'{metric}_diff'] = abs(df[metric] - target[metric])
            
        # Calculate overall similarity score (lower is more similar)
        df['similarity_score'] = sum(df[f'{metric}_diff'] for metric in metrics)
        
        # Remove the target player
        df = df[df['web_name'] != player_name]
        
        # Sort by similarity
        df = df.sort_values('similarity_score')
        
        # Select columns to display
        display_cols = ['web_name', 'team_name', 'estimated_value_millions',
                       'goals', 'assists', 'goals_per_90', 'assists_per_90', 
                       'performance_score']
        
        return df[display_cols].head(top_n)
    
    def recommend_transfers(self, target_positions: List[str], budget: float = 100.0):
        """Recommend transfer targets based on positions needed and budget"""  
        print(f"\nFinding transfer targets for positions: {', '.join(target_positions)}")
        print(f"Budget: £{budget}m")
        print("="*60)
        
        recommendations = {}
        
        for position in target_positions:
            print(f"\n{position} Options:")
            print("-"*40)
            
            # Get players in position within budget
            candidates = self.players_df[
                (self.players_df['position'] == position) &
                (self.players_df['estimated_value_millions'] <= budget) &
                (self.players_df['minutes_played'] >= 900)
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
                
                print(f"{rec['name']} ({rec['team']}) - £{rec['value']}m")
                print(f"  Goals: {rec['goals']}, Assists: {rec['assists']}")
                print(f"  Per 90: {rec['goals_per_90']}g, {rec['assists_per_90']}a")
                print(f"  Performance Score: {rec['performance_score']}")
                print()
                
        return recommendations
    
    def display_market_analysis(self):
        """Display comprehensive market analysis"""
        print("\n" + "="*60)
        print("PLAYER MARKET VALUE ANALYSIS")
        print("="*60)
        
        # Best performers by position
        positions = ['GKP', 'DEF', 'MID', 'FWD']
        
        for pos in positions:
            print(f"\n{pos} - Top Performers:")
            print("-"*60)
            top_performers = self.players_df[
                (self.players_df['position'] == pos) & 
                (self.players_df['minutes_played'] >= 900)
            ].nlargest(5, 'performance_score')
            
            display_cols = ['web_name', 'team_name', 'goals', 'assists', 
                           'performance_score', 'estimated_value_millions']
            print(top_performers[display_cols].to_string(index=False))
            
        print("\n\nMost Undervalued Players:")
        print("-"*60)
        undervalued = self.find_undervalued_players(top_n=10)
        print(undervalued.to_string(index=False))
        
        print("\n\nBest Young Prospects (High performance, lower values):")
        print("-"*60)
        prospects = self.players_df[
            (self.players_df['minutes_played'] >= 900) &
            (self.players_df['estimated_value_millions'] <= 30)
        ].nlargest(10, 'performance_score')
        
        display_cols = ['web_name', 'team_name', 'position', 'goals', 'assists',
                       'performance_score', 'estimated_value_millions']
        print(prospects[display_cols].to_string(index=False))


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TransferValueAnalyzer()
    
    # Fetch latest data
    analyzer.fetch_data()
    
    # Calculate performance metrics
    analyzer.calculate_performance_metrics()
    
    # Estimate market values
    analyzer.estimate_market_value()
    
    # Display market analysis
    analyzer.display_market_analysis()
    
    # Create visualizations
    print("\n\nGenerating visualizations...")
    
    # 1. Top performers visualization
    analyzer.visualize_top_performers()
    
    # 2. Value analysis visualization
    analyzer.visualize_value_analysis()
    
    # 3. Team analysis
    analyzer.visualize_team_analysis()
    
    # 4. Player comparison example
    # Compare specific players (uncomment and modify names)
    # analyzer.visualize_player_comparison(['Salah', 'De Bruyne', 'Fernandes'])
    
    # 5. Transfer target matrix
    analyzer.create_transfer_target_matrix(['MID', 'FWD'], max_value=40)
    
    # Example: Find similar players to a specific player
    print("\n\nSIMILAR PLAYER ANALYSIS")
    print("="*60)
    # Uncomment and replace with actual player name
    # similar = analyzer.find_similar_players("Salah", top_n=5)
    # if similar is not None:
    #     print(similar.to_string(index=False))
    
    # Example: Get transfer recommendations for specific positions
    print("\n\nTRANSFER RECOMMENDATIONS")
    print("="*60)
    recommendations = analyzer.recommend_transfers(
        target_positions=['MID', 'FWD'],
        budget=50.0  # £50m budget per player
    )