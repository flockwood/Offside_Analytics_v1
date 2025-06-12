# enhanced_main.py
"""Enhanced main application with Transfermarkt integration"""

import argparse
import os
from dotenv import load_dotenv
from enhanced_transfer_analyzer import EnhancedTransferAnalyzer

# Load environment variables
load_dotenv()


def main():
    """Enhanced main function with Transfermarkt integration"""
    parser = argparse.ArgumentParser(description='Enhanced Soccer Transfer Analyzer')
    parser.add_argument('--command', type=str, default='analyze',
                       choices=['analyze', 'discrepancies', 'contracts', 'injuries', 
                               'comprehensive', 'export'],
                       help='Command to execute')
    parser.add_argument('--player-urls', nargs='+', help='Transfermarkt player URLs')
    parser.add_argument('--injury-url', type=str, 
                       default='https://www.transfermarkt.com/premier-league/verletztespieler/wettbewerb/GB1',
                       help='Transfermarkt injury list URL')
    parser.add_argument('--league-url', type=str,
                       help='Transfermarkt league URL for comparison')
    parser.add_argument('--output', type=str, help='Output filename for exports')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    # Initialize enhanced analyzer
    api_token = os.environ.get('APIFY_API_TOKEN')
    if not api_token:
        print("Error: APIFY_API_TOKEN not found in environment variables")
        print("Please set it in your .env file or environment")
        return
    
    analyzer = EnhancedTransferAnalyzer(api_token)
    analyzer.initialize()
    
    # Execute command
    if args.command == 'analyze':
        if not args.player_urls:
            print("Error: --player-urls required for analysis")
            print("Example: --player-urls https://www.transfermarkt.com/mohamed-salah/profil/spieler/148455")
            return
        
        report = analyzer.generate_comprehensive_report(args.player_urls)
        print(report)
    
    elif args.command == 'discrepancies':
        if args.player_urls:
            analyzer.fetch_transfermarkt_data(args.player_urls)
        
        discrepancies = analyzer.analyze_value_discrepancies()
        print("\nValue Discrepancies Analysis:")
        print("="*80)
        print(discrepancies.to_string(index=False))
    
    elif args.command == 'contracts':
        if args.player_urls:
            analyzer.fetch_transfermarkt_data(args.player_urls)
        
        contracts = analyzer.find_contract_opportunities()
        print("\nContract Expiry Opportunities:")
        print("="*80)
        print(contracts.to_string(index=False))
    
    elif args.command == 'injuries':
        injuries = analyzer.analyze_injuries_impact(args.injury_url)
        
        print("\nCurrent Injuries & Replacement Suggestions:")
        print("="*80)
        for injury in injuries['injured_players']:
            print(f"\n{injury['name']} ({injury['team']}) - {injury['injury']}")
            print(f"Expected return: {injury['return']}")
            print(f"Performance score: {injury['performance_score']:.1f}")
            
            if injury['name'] in injuries['replacement_suggestions']:
                print("\nSuggested replacements:")
                for i, replacement in enumerate(injuries['replacement_suggestions'][injury['name']][:3], 1):
                    print(f"{i}. {replacement['web_name']} ({replacement['team_name']})")
    
    elif args.command == 'comprehensive':
        if not args.player_urls:
            print("Error: --player-urls required for comprehensive analysis")
            return
        
        report = analyzer.generate_comprehensive_report(args.player_urls)
        print(report)
        
        if not args.no_viz:
            # Generate visualizations
            analyzer.visualizer.plot_value_analysis(analyzer.combined_data, analyzer.player_analyzer)
            analyzer.visualizer.plot_top_performers(analyzer.combined_data)
    
    elif args.command == 'export':
        filename = args.output or 'enhanced_player_data.csv'
        if analyzer.combined_data is not None:
            analyzer.combined_data.to_csv(filename, index=False)
            print(f"Data exported to {filename}")
        else:
            print("No combined data available. Run analysis first.")


if __name__ == "__main__":
    main()