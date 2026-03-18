import os, sys
from nlp.imit_main import imit_signal as imit
from nlp.nlp_main import nlp_signal as nlp # nlp-project==0.0.9
from defirl.rl import RLmodel_bids as bid
# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Only go up one level since we're already in 'bid'
sys.path.append(project_root)
import pandas as pd
import talib.abstract as ta
import numpy as np
import logging
#from datahandler.fetcher import Fetcher
#from datahandler.fetcher import main as sig_gen

def compute_nlp_indicators(dataframe: pd.DataFrame = None):
    long_k_period = 15
    long_d_period = 3


    short_k_period = 14
    short_d_period = 3

    try:
        df = dataframe
        if df.empty:
            raise ValueError("Fetched data is empty.")

        # Compute indicators
        df['sma-05'] = ta.SMA(df['close'], timeperiod=5)
        df['sma-07'] = ta.SMA(df['close'], timeperiod=7)
        df['sma-25'] = ta.SMA(df['close'], timeperiod=25)
        df['sma-compare'] = ((df['sma-07'] > df['sma-05']) & (df['sma-25'] > df['sma-07'])).astype(int)
        df['ask'] = df['close'] * df['volume'] / (df['close'] + df['open'])
        df['bid'] = df['close'] * df['volume'] / (df['close'] + df['open'])
        df['ema-26'] = ta.EMA(df['close'], timeperiod=12)
        df['ema-12'] = ta.EMA(df['close'], timeperiod=26)
        df['macd'] = df['ema-12'] - df['ema-26']
        df['macdsignal'] = ta.EMA(df['macd'], timeperiod=9)
        df['macd-histogram'] = df['macd'] - df['macdsignal']
        df['grad-histogram'] = np.gradient(df['macd-histogram'].rolling(center=False, window=2).mean())
        df['mean-grad-hist'] = (df['grad-histogram'] > df['grad-histogram'].mean()).astype(int)
        df['long_k'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=long_k_period, slowk_period=long_d_period, slowd_period=long_d_period)[0]
        df['long_d'] = ta.SMA(df['long_k'], timeperiod=long_d_period)
        df['long_j'] = 3 * df['long_d'] - 2 * df['long_k']
        df['long_jcrosk'] = ((df['long_j'].shift(1) < df['long_k'].shift(1)) & (df['long_j'] > df['long_k'])).astype(int)
        df['short_k'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=short_k_period, slowk_period=short_d_period, slowd_period=short_d_period)[0]
        df['short_d'] = ta.SMA(df['short_k'], timeperiod=short_d_period)
        df['short_j'] = 3 * df['short_d'] - 2 * df['short_k']
        df['short_kdj_ratio'] = df['short_d'] / df['short_j']
        df['short_kdj'] = ((df['short_d'] < df['short_k']) & (df['short_k'] < df['short_j'])).astype(int)

        return df

    except Exception as e:
        logging.error(f"Error computing indicators: {e}")
        return pd.DataFrame(columns=['ask', 'bid', 'sma-compare', 'close', 'high', 'low', 'volume', 'open', 'is_short'])


#------------------------------------------------------------------------------------------------------------
""""
def getBidsig(is_short: int, ml_candle: int, dataframe: pd.DataFrame = df):
    if len(dataframe) <= ml_candle:
        logging.warning(f"Not enough data (len={len(dataframe)}) for ml_candle={ml_candle}")
        return 'do_nothing'
    
    try:
        row_data = dataframe.iloc[-ml_candle]
        model_output = bid(
            row_data['ask'],
            row_data['bid'],
            row_data['sma-compare'],
            is_short
        ).predict_action().get("action") # get('trans_action')
        return model_output
    except Exception as e:
        logging.error(f"Error in getBidsig: {str(e)}")
        return 'do_nothing'
#------------------------------------------------------------------------------------------------------------
"""
def getBidsig(is_short: int, ml_candle: int, dataframe: pd.DataFrame = None):
    if len(dataframe) <= ml_candle:
        logging.warning(f"Not enough data (len={len(dataframe)}) for ml_candle={ml_candle}")
        return None
    
    try:
        row_data = dataframe.iloc[-ml_candle]
        model_output = bid(
            row_data['ask'],
            row_data['bid'],
            row_data['sma-compare'],
            is_short
        )
        return model_output
    except Exception as e:
        logging.error(f"Error in getBidsig: {str(e)}")
        return None

def getNlpsig(ml_candle: int, dataframe: pd.DataFrame = None):
    if len(dataframe) <= ml_candle:
        logging.warning(f"Not enough data (len={len(dataframe)}) for ml_candle={ml_candle}")
        return None
    
    try:
        df = compute_nlp_indicators(dataframe)
        row_data = df.iloc[-ml_candle]
        elems = [
            row_data['open'],
            row_data['high'],
            row_data['ema-26'],
            row_data['ema-12'],
            row_data['low'],
            row_data['mean-grad-hist'],
            row_data['close'], 
            row_data['volume'], 
            row_data['sma-25'],
            row_data['long_jcrosk'],
            row_data['short_kdj']
        ]

        df[f'buy_imit_short'] = imit(*elems)
        model_output = nlp(df['buy_imit_short'].iloc[-3:].values)
        return model_output

    except Exception as e:
        logging.error(f"Error in getBidsig: {str(e)}")
        return None


"""
Example usage
print(Fetcher('ADA/USDT:USDT', 'bybit').compute_nlp_action())
print(sig_gen('ADA/USDT:USDT', 'bybit', method='nlp'))
print(sig_gen('ADA/USDT:USDT', 'bybit', method='bidrl', is_short=0).predict_action())
print(sig_gen('ADA/USDT:USDT', 'bybit', method='largerl', is_short=0).predict_action().get("action"))"
print(sig_gen('ADA/USDT:USDT', 'bybit', method='smallrl', is_short=0, ml_candle=10).predict_action().get("action"))"
"""

if __name__ == '__main__':
    compute_nlp_indicators()