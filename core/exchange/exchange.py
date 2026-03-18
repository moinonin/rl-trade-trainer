import ccxt, os
from dotenv import load_dotenv

load_dotenv('.env')
# Binance API credentials

API_KEY = os.getenv('BYBIT_TESTNET_API_KEY')
API_SECRET = os.getenv('BYBIT_TESTNET_API_SECRET')

def init_exchange(exchange_name: str, api_key: str = None, api_secret: str = None, testnet: bool = False, default_type: str = 'future'):
    """
    Initialize an exchange instance with optional API keys, testnet, and default type.
    
    :param exchange_name: Name of the exchange (e.g., 'bybit', 'binance').
    :param api_key: API key for the exchange (optional).
    :param api_secret: API secret for the exchange (optional).
    :param testnet: Whether to use the testnet (default: False).
    :param default_type: Default trading type ('spot' or 'future', default: 'future').
    :return: Initialized exchange instance.
    """
    # Get the exchange class dynamically
    exchange_class = getattr(ccxt, exchange_name)
    
    # Base configuration
    config = {
        'enableRateLimit': True,
        'options': {
            'defaultType': default_type,
        },
    }
    
    # Add API keys if provided
    if api_key and api_secret:
        config['apiKey'] = API_KEY
        config['secret'] = API_SECRET
    
    # Add testnet URLs if required
    if testnet and exchange_name == 'bybit':
        config['urls'] = {
            'api': {
                'public': 'https://api-testnet.bybit.com',
                'private': 'https://api-testnet.bybit.com',
            },
        }
    
    # Initialize the exchange
    exchange = exchange_class(config)
    return exchange