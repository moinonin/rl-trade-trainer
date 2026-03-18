import pandas as pd
import numpy as np
import logging
from nlp.imit_main import imit_signal as imit
from nlp.nlp_main import nlp_signal as nlp # nlp-project==0.1.1
from core.data.fetcher import Fetcher
from core.data.fetcher import buy_imit_short

logger = logging.getLogger("SignalLogger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_pairs(exchange):
    """Get list of available pairs from exchange"""
    try:
        markets = exchange.load_markets()
        return list(markets.keys())
    except Exception as e:
        logging.error(f"Error getting pairs: {e}")
        return []

def compute_nlp_action(pair, exchange_name, timeframe='1m', limit=30):
    """Compute NLP action for a given pair"""
    try:
        fetcher = Fetcher(pair, exchange_name, timeframe, limit)
        return fetcher.compute_nlp_action()
    except Exception as e:
        logging.error(f"Error computing NLP action: {e}")
        return {'error': str(e)}

def console_stream(pair, exchange_name, timeframe='1m', limit=30):
    fetcher = Fetcher(pair, exchange_name, timeframe, limit)
    exchange = fetcher.client
    pairs = get_pairs(exchange)
    try:
        while True:
            for pair in pairs:
                signal = compute_nlp_action(pair=pair, exchange_name=exchange_name, timeframe=timeframe, limit=limit)
                logging.info(f"{signal} signal found for {pair} on a {timeframe} timeframe ...")

    except KeyboardInterrupt:
        print("Loop interrupted by user.")


# Uncomment to run the console stream
# console_stream('btc/usdt', 'bybit', '1m', 30)
print(compute_nlp_action('btc/usdt', 'binance'))
# Example usage
''''
if __name__ == "__main__":
    result = compute_nlp_action('btc/usdt', 'bybit')
    print(result)
'''
