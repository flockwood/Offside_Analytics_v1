# config.py
"""Configuration settings for the transfer analyzer"""

API_CONFIG = {
    'base_url': 'https://fantasy.premierleague.com/api',
    'endpoints': {
        'bootstrap': '/bootstrap-static/',
        'fixtures': '/fixtures/',
        'player_detail': '/element-summary/{player_id}/'
    }
}

POSITION_MAPPING = {
    1: 'GKP',
    2: 'DEF', 
    3: 'MID',
    4: 'FWD'
}

BASE_VALUES = {
    'GKP': 10,
    'DEF': 15,
    'MID': 20,
    'FWD': 25
}

POSITION_WEIGHTS = {
    'GKP': {'clean_sheets': 1.0, 'saves': 0.5, 'goals': 0.1, 'assists': 0.1},
    'DEF': {'clean_sheets': 0.7, 'goals': 0.8, 'assists': 0.6, 'tackles': 0.5},
    'MID': {'goals': 0.9, 'assists': 1.0, 'key_passes': 0.7, 'chances': 0.6},
    'FWD': {'goals': 1.0, 'assists': 0.7, 'shots': 0.5, 'xg': 0.8}
}

VISUALIZATION_STYLE = {
    'figure_size': (15, 12),
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'husl'
}