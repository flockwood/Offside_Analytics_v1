# main.py
"""Main application file that brings all modules together"""

import argparse
import matplotlib.pyplot as plt
from data_fetcher import DataFetcher
from performance_calculator import PerformanceCalculator
from value_estimator import ValueEstimator
from player_analyzer import PlayerAnalyzer
from visualizer import Visualizer
from report_generator import ReportGenerator


class TransferAnalyzer:
    """Main application class that coordinates all modules"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.performance_calculator = PerformanceCalculator()
        self.value_estimator = ValueEstimator()
        self.visualizer = Visualizer()
        self.players_df = None
        self.player_analyzer = None
        self.report_generator = None
    
    def initialize(self):
        """Initialize the analyzer with fresh data"""
        print("Initializing Transfer Analyzer...")
        
        # Fetch data
        print("Fetching player data...")
        self.players_df = self.data_fetcher.get_players_dataframe()
        
        # Calculate metrics
        print("Calculating performance metrics...")
        self.players_df = self.performance_calculator.calculate_per_90_metrics(self.players_df)
        self.players_df = self.performance_calculator.calculate_performance_scores(self.players_df)
        self.players_df = self.performance_calculator.calculate_form_metrics(self.players_df)
        
        # Estimate values
        print("Estimating market values...")
        self.players_df = self.value_estimator.estimate_market_values(self.players_df)
        self.players_df = self.value_estimator.calculate_value_metrics(self.players_df)
        
        # Initialize analyzers
        self.player_analyzer = PlayerAnalyzer(self.players_df)
        self.report_generator = ReportGenerator(self.players_df)
        
        print("Initialization complete!")
    
    def find_undervalued_players(self, position=None, max_price=None, top_n=10):
        """Find undervalued players"""
        return self.player_analyzer.find_undervalued_players(
            position=position, max_price=max_price, top_n=top_n
        )
    
    def find_similar_players(self, player_name, top_n=5):
        """Find similar players"""
        return self.player_analyzer.find_similar_players(player_name, top_n=top_n)
    
    def recommend_transfers(self, positions, budget=50.0):
        """Get transfer recommendations"""
        return self.player_analyzer.recommend_transfers(positions, budget)
    
    def visualize_all(self):
        """Create all visualizations"""
        print("\nGenerating visualizations...")
        
        # Top performers
        self.visualizer.plot_top_performers(self.players_df)
        
        # Value analysis
        self.visualizer.plot_value_analysis(self.players_df, self.player_analyzer)
        
        # Team analysis
        self.visualizer.plot_team_analysis(self.players_df)
        
        plt.show()
    
    def generate_report(self, report_type='market'):
        """Generate text report"""
        if report_type == 'market':
            return self.report_generator.generate_market_summary()
        else:
            return "Invalid report type"
    
    def compare_players(self, player_names):
        """Compare multiple players visually"""
        return self.visualizer.plot_player_radar(self.players_df, player_names)
    
    def analyze_team(self, team_name):
        """Analyze a specific team"""
        return self.report_generator.generate_team_report(team_name)
    
    def export_data(self, filename, data_type='all'):
        """Export data to CSV"""
        self.report_generator.export_to_csv(filename, data_type)
        print(f"Data exported to {filename}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='Soccer Transfer Value Analyzer')
    parser.add_argument('--command', type=str, default='analyze',
                       choices=['analyze', 'undervalued', 'similar', 'transfers', 'compare', 'team', 'export'],
                       help='Command to execute')
    parser.add_argument('--position', type=str, help='Filter by position (GKP, DEF, MID, FWD)')
    parser.add_argument('--max-price', type=float, help='Maximum price filter')
    parser.add_argument('--player', type=str, help='Player name for similarity search')
    parser.add_argument('--players', nargs='+', help='Players to compare')
    parser.add_argument('--team', type=str, help='Team name for analysis')
    parser.add_argument('--budget', type=float, default=50.0, help='Transfer budget')
    parser.add_argument('--positions', nargs='+', help='Positions for transfer recommendations')
    parser.add_argument('--output', type=str, help='Output filename for exports')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TransferAnalyzer()
    analyzer.initialize()
    
    # Execute command
    if args.command == 'analyze':
        # Print market summary
        print(analyzer.generate_report('market'))
        
        # Show visualizations unless disabled
        if not args.no_viz:
            analyzer.visualize_all()
    
    elif args.command == 'undervalued':
        result = analyzer.find_undervalued_players(
            position=args.position,
            max_price=args.max_price
        )
        print("\nMost Undervalued Players:")
        print("="*60)
        print(result.to_string(index=False))
    
    elif args.command == 'similar':
        if not args.player:
            print("Error: --player name required for similarity search")
            return
        
        result = analyzer.find_similar_players(args.player)
        if result is not None:
            print(f"\nPlayers Similar to {args.player}:")
            print("="*60)
            print(result.to_string(index=False))
    
    elif args.command == 'transfers':
        positions = args.positions or ['MID', 'FWD']
        recommendations = analyzer.recommend_transfers(positions, args.budget)
        
        print("\nTransfer Recommendations:")
        print("="*60)
        for position, players in recommendations.items():
            print(f"\n{position} Options:")
            for i, player in enumerate(players, 1):
                print(f"{i}. {player['name']} ({player['team']}) - £{player['value']}m")
                print(f"   Performance Score: {player['performance_score']}")
    
    elif args.command == 'compare':
        if not args.players:
            print("Error: --players names required for comparison")
            return
        
        fig = analyzer.compare_players(args.players)
        if fig:
            plt.show()
    
    elif args.command == 'team':
        if not args.team:
            print("Error: --team name required for team analysis")
            return
        
        print(analyzer.analyze_team(args.team))
    
    elif args.command == 'export':
        filename = args.output or 'player_data.csv'
        analyzer.export_data(filename)


if __name__ == "__main__":
    main()